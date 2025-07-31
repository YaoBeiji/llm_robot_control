import sounddevice as sd
import numpy as np
from scipy.signal import resample
import queue
import torch
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from funasr_onnx import CT_Transformer
import numpy as np
from scipy.signal import resample
# from pynput import keyboard
import torch
from torch import nn
from funasr.models.ctc.ctc import CTC
from funasr.utils.datadir_writer import DatadirWriter
from funasr.train_utils.device_funcs import force_gatherable
from funasr.metrics.compute_acc import th_accuracy
from funasr.losses.label_smoothing_loss import LabelSmoothingLoss
from funasr.register import tables
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from funasr_onnx import CT_Transformer
import time
import webrtcvad


class SenseVoiceSmall(nn.Module):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
            self,
            specaug: str = None,
            specaug_conf: dict = None,
            normalize: str = None,
            normalize_conf: dict = None,
            encoder: str = None,
            encoder_conf: dict = None,
            ctc_conf: dict = None,
            input_size: int = 80,
            vocab_size: int = -1,
            ignore_id: int = -1,
            blank_id: int = 0,
            sos: int = 1,
            eos: int = 2,
            length_normalized_loss: bool = False,
            **kwargs,
    ):

        super().__init__()

        if specaug is not None:
            specaug_class = tables.specaug_classes.get(specaug)
            specaug = specaug_class(**specaug_conf)
        if normalize is not None:
            normalize_class = tables.normalize_classes.get(normalize)
            normalize = normalize_class(**normalize_conf)
        encoder_class = tables.encoder_classes.get(encoder)
        encoder = encoder_class(input_size=input_size, **encoder_conf)
        encoder_output_size = encoder.output_size()

        if ctc_conf is None:
            ctc_conf = {}
        ctc = CTC(odim=vocab_size, encoder_output_size=encoder_output_size, **ctc_conf)

        self.blank_id = blank_id
        self.sos = sos if sos is not None else vocab_size - 1
        self.eos = eos if eos is not None else vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.specaug = specaug
        self.normalize = normalize
        self.encoder = encoder
        self.error_calculator = None

        self.ctc = ctc

        self.length_normalized_loss = length_normalized_loss
        self.encoder_output_size = encoder_output_size

        self.lid_dict = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
        self.lid_int_dict = {24884: 3, 24885: 4, 24888: 7, 24892: 11, 24896: 12, 24992: 13}
        self.textnorm_dict = {"withitn": 14, "woitn": 15}
        self.textnorm_int_dict = {25016: 14, 25017: 15}
        self.embed = torch.nn.Embedding(7 + len(self.lid_dict) + len(self.textnorm_dict), input_size)
        self.emo_dict = {"unk": 25009, "happy": 25001, "sad": 25002, "angry": 25003, "neutral": 25004}

        self.criterion_att = LabelSmoothingLoss(
            size=self.vocab_size,
            padding_idx=self.ignore_id,
            smoothing=kwargs.get("lsm_weight", 0.0),
            normalize_length=self.length_normalized_loss,
        )

    @staticmethod
    def from_pretrained(model: str = None, **kwargs):
        from funasr import AutoModel
        model, kwargs = AutoModel.build_model(model=model, trust_remote_code=True, **kwargs)

        return model, kwargs

    def forward(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
            **kwargs,
    ):
        """Encoder + Decoder + Calc loss
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                text: (Batch, Length)
                text_lengths: (Batch,)
        """
        # import pdb;
        # pdb.set_trace()
        if len(text_lengths.size()) > 1:
            text_lengths = text_lengths[:, 0]
        if len(speech_lengths.size()) > 1:
            speech_lengths = speech_lengths[:, 0]

        batch_size = speech.shape[0]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths, text)

        loss_ctc, cer_ctc = None, None
        loss_rich, acc_rich = None, None
        stats = dict()

        loss_ctc, cer_ctc = self._calc_ctc_loss(
            encoder_out[:, 4:, :], encoder_out_lens - 4, text[:, 4:], text_lengths - 4
        )

        loss_rich, acc_rich = self._calc_rich_ce_loss(
            encoder_out[:, :4, :], text[:, :4]
        )

        loss = loss_ctc + loss_rich
        # Collect total loss stats
        stats["loss_ctc"] = torch.clone(loss_ctc.detach()) if loss_ctc is not None else None
        stats["loss_rich"] = torch.clone(loss_rich.detach()) if loss_rich is not None else None
        stats["loss"] = torch.clone(loss.detach()) if loss is not None else None
        stats["acc_rich"] = acc_rich

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:
            batch_size = int((text_lengths + 1).sum())
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def encode(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            text: torch.Tensor,
            **kwargs,
    ):
        """Frontend + Encoder. Note that this method is used by asr_inference.py
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                ind: int
        """

        # Data augmentation
        if self.specaug is not None and self.training:
            speech, speech_lengths = self.specaug(speech, speech_lengths)

        # Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
        if self.normalize is not None:
            speech, speech_lengths = self.normalize(speech, speech_lengths)

        lids = torch.LongTensor(
            [[self.lid_int_dict[int(lid)] if torch.rand(1) > 0.2 and int(lid) in self.lid_int_dict else 0] for lid in
             text[:, 0]]).to(speech.device)
        language_query = self.embed(lids)

        styles = torch.LongTensor([[self.textnorm_int_dict[int(style)]] for style in text[:, 3]]).to(speech.device)
        style_query = self.embed(styles)
        speech = torch.cat((style_query, speech), dim=1)
        speech_lengths += 1

        event_emo_query = self.embed(torch.LongTensor([[1, 2]]).to(speech.device)).repeat(speech.size(0), 1, 1)
        input_query = torch.cat((language_query, event_emo_query), dim=1)
        speech = torch.cat((input_query, speech), dim=1)
        speech_lengths += 3

        encoder_out, encoder_out_lens = self.encoder(speech, speech_lengths)

        return encoder_out, encoder_out_lens

    def _calc_ctc_loss(
            self,
            encoder_out: torch.Tensor,
            encoder_out_lens: torch.Tensor,
            ys_pad: torch.Tensor,
            ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def _calc_rich_ce_loss(
            self,
            encoder_out: torch.Tensor,
            ys_pad: torch.Tensor,
    ):
        decoder_out = self.ctc.ctc_lo(encoder_out)
        # 2. Compute attention loss
        loss_rich = self.criterion_att(decoder_out, ys_pad.contiguous())
        acc_rich = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_pad.contiguous(),
            ignore_label=self.ignore_id,
        )

        return loss_rich, acc_rich

    def inference(
            self,
            data_in,
            data_lengths=None,
            key: list = ["wav_file_tmp_name"],
            tokenizer=None,
            frontend=None,
            **kwargs,
    ):

        meta_data = {}
        if (
                isinstance(data_in, torch.Tensor) and kwargs.get("data_type", "sound") == "fbank"
        ):  # fbank
            speech, speech_lengths = data_in, data_lengths
            if len(speech.shape) < 3:
                speech = speech[None, :, :]
            if speech_lengths is None:
                speech_lengths = speech.shape[1]
        else:
            # extract fbank feats
            time1 = time.perf_counter()
            audio_sample_list = load_audio_text_image_video(
                data_in,
                fs=frontend.fs,
                audio_fs=kwargs.get("fs", 16000),
                data_type=kwargs.get("data_type", "sound"),
                tokenizer=tokenizer,
            )
            time2 = time.perf_counter()
            meta_data["load_data"] = f"{time2 - time1:0.3f}"
            speech, speech_lengths = extract_fbank(
                audio_sample_list, data_type=kwargs.get("data_type", "sound"), frontend=frontend
            )
            time3 = time.perf_counter()
            meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
            meta_data["batch_data_time"] = (
                    speech_lengths.sum().item() * frontend.frame_shift * frontend.lfr_n / 1000
            )

        speech = speech.to(device=kwargs["device"])
        speech_lengths = speech_lengths.to(device=kwargs["device"])

        language = kwargs.get("language", "auto")
        language_query = self.embed(
            torch.LongTensor(
                [[self.lid_dict[language] if language in self.lid_dict else 0]]
            ).to(speech.device)
        ).repeat(speech.size(0), 1, 1)

        use_itn = kwargs.get("use_itn", False)
        textnorm = kwargs.get("text_norm", None)
        if textnorm is None:
            textnorm = "withitn" if use_itn else "woitn"
        textnorm_query = self.embed(
            torch.LongTensor([[self.textnorm_dict[textnorm]]]).to(speech.device)
        ).repeat(speech.size(0), 1, 1)
        speech = torch.cat((textnorm_query, speech), dim=1)
        speech_lengths += 1

        event_emo_query = self.embed(torch.LongTensor([[1, 2]]).to(speech.device)).repeat(
            speech.size(0), 1, 1
        )
        input_query = torch.cat((language_query, event_emo_query), dim=1)
        speech = torch.cat((input_query, speech), dim=1)
        speech_lengths += 3

        # Encoder
        encoder_out, encoder_out_lens = self.encoder(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        # c. Passed the encoder result and the beam search
        ctc_logits = self.ctc.log_softmax(encoder_out)
        if kwargs.get("ban_emo_unk", False):
            ctc_logits[:, :, self.emo_dict["unk"]] = -float("inf")

        results = []
        b, n, d = encoder_out.size()
        if isinstance(key[0], (list, tuple)):
            key = key[0]
        if len(key) < b:
            key = key * b
        for i in range(b):
            x = ctc_logits[i, : encoder_out_lens[i].item(), :]
            yseq = x.argmax(dim=-1)
            yseq = torch.unique_consecutive(yseq, dim=-1)

            ibest_writer = None
            if kwargs.get("output_dir") is not None:
                if not hasattr(self, "writer"):
                    self.writer = DatadirWriter(kwargs.get("output_dir"))
                ibest_writer = self.writer[f"1best_recog"]

            mask = yseq != self.blank_id
            token_int = yseq[mask].tolist()

            # Change integer-ids to tokens
            text = tokenizer.decode(token_int)

            result_i = {"key": key[i], "text": text}
            results.append(result_i)

            if ibest_writer is not None:
                ibest_writer["text"][key[i]] = text

        return results, meta_data

    def export(self, **kwargs):
        from export_meta import export_rebuild_model

        if "max_seq_len" not in kwargs:
            kwargs["max_seq_len"] = 512
        models = export_rebuild_model(model=self, **kwargs)
        return models


class VADVoiceRecorderWithPlayback:
    def __init__(self, aggressiveness=3,
                 model_path="./SenseVoiceSmall",
                 ct_model_path="./ct-transformer-large",
                 device_index=None):
        self.model, self.kwargs = SenseVoiceSmall.from_pretrained(model=model_path, device="cpu")
        self.model.eval()
        self.ct_model = CT_Transformer(ct_model_path, batch_size=1, quantize=True)
        self.vad = webrtcvad.Vad(aggressiveness)
        self.device_index = device_index
        self.sample_rate = 16000
        self.frame_duration = 30  # ms
        self.frame_size = int(self.sample_rate * self.frame_duration / 1000)  # 每帧样本数
        self.recording = False
        self.audio_buffer = bytearray()
        self.silence_limit = 2.0  # 静音超过1秒停止录音
        self.silence_start = None
        self.audio_q = queue.Queue()

    def is_speech(self, frame_bytes):
        return self.vad.is_speech(frame_bytes, self.sample_rate)

    def callback(self, indata, frames, time_info, status):
        self.audio_q.put(bytes(indata))

    def listen_and_record(self):
        print("启动VAD录音，检测到语音自动开始录音，静音停止。")
        with sd.RawInputStream(samplerate=self.sample_rate, blocksize=self.frame_size,
                               dtype='int16', channels=1,
                               callback=self.callback, device=self.device_index):
            while True:
                frame = self.audio_q.get()
                if len(frame) < self.frame_size * 2:
                    continue
                speech = self.is_speech(frame)
                if speech:
                    if not self.recording:
                        print("检测到语音，开始录音")
                        self.recording = True
                        self.audio_buffer = bytearray()
                    self.audio_buffer.extend(frame)
                    self.silence_start = None
                else:
                    if self.recording:
                        if self.silence_start is None:
                            self.silence_start = time.time()
                        elif time.time() - self.silence_start > self.silence_limit:
                            print("静音超过阈值，停止录音，准备播放录音")
                            self.recording = False
                            # self.play_audio()
                            self.inf()
                            self.audio_buffer = bytearray()
                        else:
                            self.audio_buffer.extend(frame)

    def play_audio(self):
        if not self.audio_buffer:
            print("无录音数据，无法播放")
            return
        # 将字节数据转换成 numpy int16 数组
        audio_np = np.frombuffer(self.audio_buffer, dtype=np.int16)
        print(f"播放录音，长度 {len(audio_np) / self.sample_rate:.2f} 秒")
        sd.play(audio_np, samplerate=self.sample_rate)
        sd.wait()  # 等待播放完毕

    def inf(self):
        audio_np = np.frombuffer(self.audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
        res = self.model.inference(
            # data_in=self.audio_buffer,
            data_in=audio_np,
            fs=self.sample_rate,
            language="auto",
            use_itn=False,
            ban_emo_unk=True,
            **self.kwargs,
        )
        print("原始识别结果:", res)
        response = rich_transcription_postprocess(res[0][0]["text"])
        output_str = self.ct_model(text=response)[0] if response else ""
        print(f"识别输出: {output_str}")
        self.audio_buffer = np.zeros((0,), dtype=np.float32)


if __name__ == "__main__":
    # device_index可以通过 `sd.query_devices()` 查看设备索引，None用默认设备
    recorder = VADVoiceRecorderWithPlayback(device_index=None)
    try:
        recorder.listen_and_record()
    except KeyboardInterrupt:
        print("程序退出")