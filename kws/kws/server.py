import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
import numpy as np
import bpu_infer_lib
import paddle
import paddleaudio
from paddleaudio.compliance.kaldi import fbank


THRES = 60000
feat_func = lambda waveform, sr: fbank(
    waveform=paddle.to_tensor(waveform), 
    sr=sr, 
    frame_shift=10, 
    frame_length=25, 
    n_mels=80)

def audio_trunc(audio_arr, thres=THRES):
    length = audio_arr.shape[1]
    if length > thres:
        audio_arr = audio_arr[:, :thres]
        return audio_arr
    elif length < thres:
        pad_zero = paddle.zeros((1,THRES), dtype=audio_arr.dtype)
        pad_zero[:, :length] = audio_arr
        return pad_zero


class KWSServer(Node):
    def __init__(self):
        super().__init__('kws_server')
        self.declare_parameter('model_path', 'kws.bin')
        self.declare_parameter('wav_path', 'keyword.wav')
        self.srv = self.create_service(Trigger, 'start_kws', self.start_kws_callback)
        self.get_logger().info("KWS Server is ready.")

    def start_kws_callback(self, request, response):
        model_path = self.get_parameter('model_path').value
        wav_path = self.get_parameter('wav_path').value
        self.get_logger().info(f"Using model path: {model_path}")
        
        inf = bpu_infer_lib.Infer(False)
        inf.load_model(model_path)
        
        key_test_load = paddleaudio.load(wav_path)
        key_test_load = (audio_trunc(key_test_load[0]), key_test_load[1])
        keyword_feat = feat_func(*key_test_load)
        key_input = keyword_feat.unsqueeze(0).numpy()
        inf.read_input(key_input, 0)
        inf.forward()
        inf.get_output()
        out = inf.outputs[0].data
        keyword_score = np.max(out).item()
        
        response.success = True
        response.message = f"The keyword matchness score is: {keyword_score}"
        return response


def main(args=None):
    rclpy.init(args=args)
    node = KWSServer()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()