import pyaudio
import wave
import threading
import numpy as np
from scipy.signal import resample
from pynput import keyboard
# import keyboard
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
    def from_pretrained(model:str=None, **kwargs):
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


        lids = torch.LongTensor([[self.lid_int_dict[int(lid)] if torch.rand(1) > 0.2 and int(lid) in self.lid_int_dict else 0 ] for lid in text[:, 0]]).to(speech.device)
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


class VoiceRecorder:
    def __init__(self,
             model_path="./SenseVoiceSmall",
             ct_model_path="./weights/ct-transformer-large",
             device_index=0,
            #  output_wav="./record/output.wav"
             ):

        # 初始化模型
        self.model, self.kwargs = SenseVoiceSmall.from_pretrained(model=model_path, device="cpu")
        self.model.eval()
        self.ct_model = CT_Transformer(ct_model_path, batch_size=1, quantize=True)
        # PyAudio相关配置
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.CHUNK = 1024
        self.DEVICE_INDEX = device_index
        self.p = pyaudio.PyAudio()
        self.device_info = self.p.get_device_info_by_index(self.DEVICE_INDEX)
        self.DEVICE_SAMPLE_RATE = int(self.device_info["defaultSampleRate"])
        self.TARGET_SAMPLE_RATE = 16000
        # self.output_wav = output_wav
        # 录音状态
        self.recording = False
        self.audio_buffer = np.zeros((0,), dtype=np.float32)
        self.stream = None

    def start_recording(self):
        if not self.recording:
            print("开始录音... (Ctrl 按下)")
            self.recording = True
            self.audio_buffer = np.zeros((0,), dtype=np.float32)
            self.stream = self.p.open(format=self.FORMAT,
                                      channels=self.CHANNELS,
                                      rate=self.DEVICE_SAMPLE_RATE,
                                      input=True,
                                      input_device_index=self.DEVICE_INDEX,
                                      frames_per_buffer=self.CHUNK)
            record_thread = threading.Thread(target=self.record_audio)
            record_thread.start()

    def stop_recording(self):
        if self.recording:
            print("停止录音，开始推理 (Alt 按下)")
            self.recording = False
            self.stream.stop_stream()
            self.stream.close()
            # self.save_audio()

            if len(self.audio_buffer) == 0:
                print("缓冲为空，跳过推理。")
                return

            print("推理中...")
            res = self.model.inference(
                data_in=self.audio_buffer,
                fs=self.TARGET_SAMPLE_RATE,
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

    def record_audio(self):
        while self.recording and self.stream.is_active():
            try:
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                if self.DEVICE_SAMPLE_RATE != self.TARGET_SAMPLE_RATE:
                    new_len = int(len(audio_np) * self.TARGET_SAMPLE_RATE / self.DEVICE_SAMPLE_RATE)
                    audio_np = resample(audio_np, new_len)
                self.audio_buffer = np.concatenate((self.audio_buffer, audio_np))
            except Exception as e:
                print(f"[录音线程异常] {e}")
                break

    def save_audio(self):
        audio_int16 = (self.audio_buffer * 32768).astype(np.int16)
        wf = wave.open(self.output_wav, "wb")
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        wf.setframerate(self.TARGET_SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())
        wf.close()
        print(f"已保存音频到 {self.output_wav}")

    # pynput 监听 需图形化界面
    def listen_hotkeys(self):
        def on_press(key):
            try:
                if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
                    self.start_recording()
                elif key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
                    self.stop_recording()
                elif key == keyboard.Key.esc:
                    print("ESC 检测到，退出程序")
                    return False
            except Exception as e:
                print("键盘监听异常:", e)
    # keyboard 监听 需root权限
    # def listen_hotkeys(self):
    #     print("监听已启动：按 Ctrl 开始录音，Alt 停止识别，ESC 退出程序。")
    #     keyboard.add_hotkey('ctrl', self.start_recording)
    #     keyboard.add_hotkey('alt', self.stop_recording)
    #     keyboard.wait('esc')
    #     print("ESC 检测到，退出程序")

    #     print("监听已启动：按 Ctrl 开始录音，Alt 停止识别，ESC 退出程序。")
    #     with keyboard.Listener(on_press=on_press) as listener:
    #         listener.join()

    def terminate(self):
        self.p.terminate()
        print("程序退出")

def record(device_index = 0):
    recorder = VoiceRecorder(device_index = 0)
    try:
        recorder.listen_hotkeys()
    finally:
        recorder.terminate()

if __name__ == "__main__":
    record(device_index = 0) # 音频设备