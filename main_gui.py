import sys  # import必要的包
import Ui_computer 
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *
from PyQt5.QtCore import *
import numpy as np
from math import pi

import wave

amp=100#阈值

cal=''#计算结果
calculation=""
recode=''
hammingcode=""

import random

def generate_expression():
    # 生成个位数
    num1 = random.randint(0, 9)
    num2 = random.randint(0, 9)
    
    # 随机选择加号或减号
    operator = random.choice(['+', '-'])
    
    # 生成加减法字符串
    expression = f"{num1}{operator}{num2}"
    
    return expression
class SimpleEncoding:
#编码器
    def __init__(self, encoding_map):
        self.encoding_map = encoding_map
 
    def encode(self, char):
        return self.encoding_map.get(char, "UNKNOWN")
 
# 定义编码映射
encoding_map = {
    '0': '00000000',
    '1': '01101001',
    '2': '10101010',
    '3': '11000011',
    '4': '11001100',
    '5': '10100101',
    '6': '01100110',
    '7': '00001111',
    '8': '11110000',
    '9': '10011001',
    '+': '01011010',
    '-': '00110011',
    '*': '00111100',
    '/': '01010101',
    '(': '10010110',
    ')': '11111111',
    
    '0000000': '0',
    '1101001': '1',
    '0101010': '2',
    '1000011': '3',
    '1001100': '4',
    '0100101': '5',
    '1100110': '6',
    '0001111': '7',
    '1110000': '8',
    '0011001': '9',
    '1011010': '+',
    '0110011': '-',
    '0111100': '*',
    '1010101': '/',
    '0010110': '(',
    '1111111': ')'
}
 
# 创建编码对象
encoder = SimpleEncoding(encoding_map)
 

# 定义编码映射
encoding_map2 = {
    '0': '0000',
    '1': '0001',
    '2': '0010',
    '3': '0011',
    '4': '0100',
    '5': '0101',
    '6': '0110',
    '7': '0111',
    '8': '1000',
    '9': '1001',
    '+': '1010',
    '-': '1011',
    '*': '1100',
    '/': '1101',
    '(': '1110',
    ')': '1111',
    
    '0000': '0',
    '0001': '1',
    '0010': '2',
    '0011': '3',
    '0100': '4',
    '0101': '5',
    '0110': '6',
    '0111': '7',
    '1000': '8',
    '1001': '9',
    '1010': '+',
    '1011': '-',
    '1100': '*',
    '1101': '/',
    '1110': '(',
    '1111': ')'
}
 
# 创建编码对象
encoder2 = SimpleEncoding(encoding_map2)
 
def clear():
    # 清空输入
    global calculation
    calculation=""
    ui.textEdit_3.clear()
    ui.textEdit_2.append("清空输入")

def clear_2():
    # 清空各个控件
    ui.textEdit_4.clear()
    ui.textEdit_2.clear()
    ui.lineEdit.clear()
    ui.lineEdit_2.clear()
    ui.lineEdit_3.clear()


have_code_flag=0# 是否有输入代码标志位
def finish():
    #输入结束
    global have_code_flag
    global calculation
    calculation=str(ui.textEdit_3.toPlainText())
    try:
        #如果能计算出结果
        eval(calculation)
        ui.textEdit_2.append("完成输入，输入的表达式是："+calculation)
        have_code_flag=1
        hamming_encode()
        send()
    except:
        #输入表达式无效
        ui.textEdit_2.append("您输入的表达式无效，请重新输入")
        clear()


import os
#音频片段
t_extra = np.linspace(0, 0.2, int(48000 * 0.2), endpoint=False)
audio_da= 32000 * np.sin(2 * np.pi * 3000 * t_extra)
def fsk(code):
    #fsk调制
    global audio_da
    duration =0.01
    sampling_freq = 48000  # 采样频率
    tone_freq1 = 2000 
    tone_freq2 = 4000 
    t = np.linspace(0, duration, int(sampling_freq * duration), endpoint=False)
    # audio_da= 32000 * np.sin(2 * np.pi * tone_freq_extra * t_extra)
    audio_data_list=[audio_da]
    # audio_data_list=[]
    for co in code:
        if(co=='0'):
            audio_data_list.append(7000 * np.sin(2 * np.pi * tone_freq1 * t))
            # audio_data = np.concatenate(audio_data,0.5 * np.sin(2 * np.pi * tone_freq1 * t))
        else:
            audio_data_list.append(32000 * np.sin(2 * np.pi * tone_freq2 * t))
            # audio_data = np.concatenate(audio_data,0.5 * np.sin(2 * np.pi * tone_freq2 * t))
    # 将生成的音频数据保存为WAV文件
    audio_data_list.append(audio_da)
    # audio_data_list.append(audio_da)
    # audio_data_list.append(audio_da)
    audio_data_combined = np.concatenate(audio_data_list)
    wavfile.write('generated_audio.wav', sampling_freq, np.int16(audio_data_combined))
    os.system('generated_audio.wav')

def re_fsk(code):
    #回传信号的fsk调制
    t_extra = np.linspace(0, 0.2, int(48000 * 0.2), endpoint=False)
    audio_d= 32000 * np.sin(2 * np.pi * 2500 * t_extra)
    duration =0.01
    sampling_freq = 48000  # 采样频率
    tone_freq1 = 2000 
    tone_freq2 = 4000 
    t = np.linspace(0, duration, int(sampling_freq * duration), endpoint=False)
    audio_data_list=[audio_d]
    # audio_data_list=[]
    for co in code:
        if(co=='0'):
            audio_data_list.append(7000 * np.sin(2 * np.pi * tone_freq1 * t))
            # audio_data = np.concatenate(audio_data,0.5 * np.sin(2 * np.pi * tone_freq1 * t))
        else:
            audio_data_list.append(32000 * np.sin(2 * np.pi * tone_freq2 * t))
            # audio_data = np.concatenate(audio_data,0.5 * np.sin(2 * np.pi * tone_freq2 * t))
    # 将生成的音频数据保存为WAV文件
    audio_data_list.append(audio_d)
    # audio_data_list.append(audio_d)
    # audio_data_list.append(audio_d)
    audio_data_combined = np.concatenate(audio_data_list)
    wavfile.write('re_generated_audio.wav', sampling_freq, np.int16(audio_data_combined))
    os.system('re_generated_audio.wav')

def send():
    #发送信号
    global hammingcode
    if(hammingcode==''):
        ui.textEdit_2.append("未编码")
    else:
        fsk(hammingcode)
        ui.textEdit_2.append("已发送")

# 将 bytes 对象转为二进制
def bytes_to_binary(data):
    decimal = int.from_bytes(data, byteorder='big')  # 将 bytes 转为十进制整数
    binary = bin(decimal)[2:]  # 将十进制整数转为二进制字符串
    formatted_binary = format(binary, '0>8')  # 格式化二进制字符串为 8 位二进制表示形式
    return formatted_binary

barker = "1111100110101"#巴克码
def hamming_encode():
    #编码
    global have_code_flag
    global calculation
    global hammingcode
    global barker
    hammingcode=""
    if(have_code_flag==0):
        ui.textEdit_2.append("您还未输入算式")
    else:
        if(ui.checkBox_3.isChecked()==True):
            hammingcode+=barker
            number=len(calculation)
            if(number>=10):
                for charn in str(number):
                    hammingcode=hammingcode+encoder.encode(charn)
            else:
                hammingcode=hammingcode+encoder.encode('0')
                hammingcode=hammingcode+encoder.encode(str(number))
            for char in calculation:
                hammingcode=hammingcode+encoder.encode(char)
            ui.textEdit_2.append("已编码，码为："+hammingcode)
        else:
            hammingcode+=barker
            number=len(calculation)
            if(number>=10):
                for charn in str(number):
                    hammingcode=hammingcode+encoder2.encode(charn)
            else:
                hammingcode=hammingcode+encoder2.encode('0')
                hammingcode=hammingcode+encoder2.encode(str(number))
            for char in calculation:
                hammingcode=hammingcode+encoder2.encode(char)
            ui.textEdit_2.append("已编码，码为："+hammingcode)

def re_encode():
    #结果回传的编码
    global cal
    global recode
    global barker
    recode=''
    recode+=barker
    if(ui.checkBox_3.isChecked()==True):
        if cal=='error':
            recode=recode+encoder.encode('0')
            recode=recode+encoder.encode('0')
        else:
            number=len(str(eval(cal)))
            if(number>=10):
                for charn in str(number):
                    recode=recode+encoder.encode(charn)
            else:
                recode=recode+encoder.encode('0')
                recode=recode+encoder.encode(str(number))
            for char in str(eval(cal)):
                recode=recode+encoder.encode(char)
        ui.textEdit_2.append("回传信息已编码，码为："+recode)
    else:
        if cal=='error':
            recode=recode+encoder2.encode('0')
            recode=recode+encoder2.encode('0')
        else:
            number=len(str(eval(cal)))
            if(number>=10):
                for charn in str(number):
                    recode=recode+encoder2.encode(charn)
            else:
                recode=recode+encoder2.encode('0')
                recode=recode+encoder2.encode(str(number))
            for char in str(eval(cal)):
                recode=recode+encoder2.encode(char)
        ui.textEdit_2.append("回传信息已编码，码为："+recode)

class Recorder(QAudioInput):
    #录音机类
    def __init__(self, format, parent=None):#初始化
        super().__init__(format, parent)
        self.temp_file = QFile("output_temp.raw")
        self.wav_file_path = "output.wav"

    def start(self):
        #开始录音
        if self.temp_file.open(QIODevice.WriteOnly):
            super().start(self.temp_file)
            ui.textEdit_2.append("开始录音")


    def stop(self):
        #结束录音
        super().stop()
        self.temp_file.close()
        self.convert_to_wav()
        ui.textEdit_2.append("结束录音")

    def convert_to_wav(self):
        #转成wav
        with open(self.temp_file.fileName(), "rb") as raw_file:
            raw_data = raw_file.read()

        with wave.open(self.wav_file_path, "wb") as wav_file:
            # 设置WAV文件的参数，这些参数应与QAudioFormat中的设置相匹配
            n_channels = self.format().channelCount()
            sample_width = self.format().sampleSize() // 8
            frame_rate = self.format().sampleRate()
            n_frames = len(raw_data) // (n_channels * sample_width)
            comp_type = "NONE"  # 无压缩
            comp_name = "not compressed"

            wav_file.setnchannels(n_channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(frame_rate)
            wav_file.setnframes(n_frames)
            wav_file.setcomptype(comp_type, comp_name)

            wav_file.writeframes(raw_data)

        # 清理临时文件
        self.temp_file.remove()

import scipy.signal as signal
from scipy.io import wavfile
import matplotlib.pyplot as plt

def on_record():
    #开始录音
    recorder.start()

def on_stop():
    #结束录音
    recorder.stop()

def read_wave(file_path):
    #读取wav文件
    with wave.open(file_path, 'r') as wav_file:
        n_channels, sampwidth, framerate, n_frames, comptype, compname = wav_file.getparams()
        frames = wav_file.readframes(n_frames)
        audio_data = np.frombuffer(frames, dtype=np.int16)
    return audio_data, framerate

def is_odd_parity_correct(bits):
    #判断奇偶性
    data_bits = bits[1:]
    count_ones = sum(data_bits)
    if count_ones % 2 == 1:
        return bits == 1
    else:
        return bits == 0

from functools import reduce
error_flag=0
def find_char(ch):
    #线性分组码的纠错过程
    global error_flag
    result=''
    if(ui.checkBox_3.isChecked()==True):
        if(sum(ch)==0):
            index=0
        else:
            index=reduce(lambda x,y:x^y,[i for i,bit in enumerate(ch) if bit])
        if (index==0):
            result=encoder.encode(''.join(str(element) for element in ch[1:]))
        elif(index!=0 and is_odd_parity_correct(ch)):
            ch[index]=~ch[index]
            result=encoder.encode(''.join(str(element) for element in ch[1:]))
        else:
            error_flag=1
            result='0'
    else:
        result=encoder2.encode(''.join(str(element) for element in ch))
    return result

from scipy.signal import correlate
def de_hammingcode():
    #译码过程
    global barker
    global error_flag
    global audio_da
    global start_record,record_time,amplitude,count,cal,recode,re_start_record
    global continue_count,right_count
    try:
    # if(1):
        error_flag==0
        barker_code=[]
        duration =0.01
        sampling_freq = 48000  # 采样频率
        tone_freq1 = 2000 
        tone_freq2 = 4000 
        t = np.linspace(0, duration, int(sampling_freq * duration), endpoint=False)
        for co in barker:
            if(co=='0'):
                barker_code.append(1 * np.sin(2 * np.pi * tone_freq1 * t))
            else:
                barker_code.append(1 * np.sin(2 * np.pi * tone_freq2 * t))
        barker_code_combined =  np.concatenate(barker_code)
        audio_data1, framerate = read_wave('output.wav')
        # audio_data1, framerate = read_wave(r'C:\Users\19144\Desktop\通信原理\generated_audio.wav')
        correlation = correlate(audio_data1, barker_code_combined, mode='same')
        max_index = np.argmax(np.abs(correlation))+3120+240
        ui.textEdit_2.append(f"信号的起始位置是"+str(max_index))
        leng=len(audio_data1)
        # 带通椭圆滤波器设计，通带为[1500，2500]
        [b11,a11] = signal.ellip(5, 0.5, 60, [1500 * 2 / sampling_freq, 2500 * 2 / sampling_freq], btype = 'bandpass', analog = False, output = 'ba')
        # 低通滤波器设计，通带截止频率为480Hz
        [b12,a12] = signal.ellip(5, 0.5, 60, (480 * 2 / sampling_freq), btype = 'lowpass', analog = False, output = 'ba')
        # 通过带通滤波器滤除带外噪声
        bandpass_out1 = signal.filtfilt(b11, a11, audio_data1)
        # 通过包络检波器
        lowpass_out1 = signal.filtfilt(b12, a12, np.abs(bandpass_out1))
        # 带通椭圆滤波器设计2，通带为[3500，4500]
        [b21,a21] = signal.ellip(5, 0.5, 60, [3500 * 2 / sampling_freq, 4500 * 2 / sampling_freq], btype = 'bandpass', analog = False, output = 'ba')
        # 通过带通滤波器滤除带外噪声
        bandpass_out2 = signal.filtfilt(b21, a21, audio_data1)
        # 通过包络检波器
        lowpass_out2 = signal.filtfilt(b12, a12, np.abs(bandpass_out2))
        
        if ui.checkBox_4.isChecked==True:
            #绘图
            fig2 = plt.figure()
            fig2.add_subplot(5, 1, 1)
            plt.plot(np.arange(0, (len(audio_data1)),1), audio_data1, 'r')
            plt.show()
            fig2.add_subplot(5, 1, 2)
            plt.plot(np.arange(0, (len(bandpass_out1)),1), bandpass_out1, 'r')
            plt.show()
            fig2.add_subplot(5, 1, 3)
            plt.plot(np.arange(0, (len(lowpass_out1)),1), lowpass_out1, 'r')
            plt.show()
            fig2.add_subplot(5, 1, 4)
            plt.plot(np.arange(0, (len(bandpass_out2)),1), bandpass_out2, 'r')
            plt.show() 
            fig2.add_subplot(5, 1, 5)
            plt.plot(np.arange(0, (len(lowpass_out2)),1), lowpass_out2, 'r')
            plt.show()
        
        binary_data=[]
        for i in range(max_index, leng, 480):
            if(lowpass_out1[i]>lowpass_out2[i]):
                binary_data.append(0)
            else:
                binary_data.append(1)
        cal=''
        if(ui.checkBox_3.isChecked()==True):
            shi=int(find_char(binary_data[0:8]))
            ge=int(find_char(binary_data[8:16]))
            num=shi*10+ge
            if(num==0 and ui.checkBox_2.isChecked()==True):
                os.system('generated_audio.wav')
                ui.textEdit_2.append("发生错误，重新传输")
            else:
                ui.textEdit_2.append("得到的码为"+str(binary_data[16:16+num*8]))
                for i in range(16, 16+num*8, 8):
                    cal+=find_char(binary_data[i:i+8])
                ui.textEdit_2.append("识别到的表达式："+cal)
                ui.textEdit_2.append("结果为"+str(eval(cal)))
                ui.lineEdit.setText(str(eval(cal)))
        else:
            shi=int(find_char(binary_data[0:4]))
            ge=int(find_char(binary_data[4:8]))
            num=shi*10+ge
            if(num==0 and ui.checkBox_2.isChecked()==True):
                os.system('generated_audio.wav')
                ui.textEdit_2.append("发生错误，重新传输")
            else:
                ui.textEdit_2.append("得到的码为"+str(binary_data[8:8+num*4]))
                for i in range(8, 8+num*4, 4):
                    cal+=find_char(binary_data[i:i+4])
                ui.textEdit_2.append("识别到的表达式："+cal)
                ui.textEdit_2.append("结果为"+str(eval(cal)))
                ui.lineEdit.setText(str(eval(cal)))
        
        if(ui.checkBox.isChecked()==True):#接受方
            re_encode()
            ui.textEdit_2.append("回传结果")
            re_fsk(recode)
        
        if(ui.checkBox.isChecked()==False and continue_flag==1):
            right_count+=1
            continue_send_handle()
        
        # start_record=0
        # re_start_record=0
        record_time=0
        amplitude=0
        count=0
    except:
        ui.lineEdit.setText("error")
        ui.textEdit_2.append("error")
        if(ui.checkBox.isChecked()==False and continue_flag==1):
            continue_send_handle()
        if(ui.checkBox_2.isChecked()==True):
            ui.textEdit_2.append("有错误，请求重传")
            cal='error'
            re_encode()
            re_fsk(recode)
            # start_record=0
            # re_start_record=0
            record_time=0
            amplitude=0
            count=0


import pyaudio
from scipy.fft import fft
import time
start_record=0
re_start_record=0
record_time=0
amplitude=0
count=0
def detect():
    #检测频率分量
    global start_record,re_start_record,record_time,amplitude,count,amp
    # if(ui.checkBox.isChecked()==True):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                                channels=1,
                                rate=48000,
                                input=True,
                                frames_per_buffer=1024)
    data = np.frombuffer(stream.read(1024), dtype=np.int16)
    fft_data = fft(data)
    freq_res = 48000 / 1024
    
    if(ui.checkBox.isChecked()==True):#接受方
        idx = int(3000 / freq_res)
        amplitude = int(np.abs(fft_data[idx]) / 10240)
        ui.textEdit_4.append("当前声音："+str(amplitude))
        if(start_record==0):
            if(amplitude>amp):
                
                record_time=time.time()
                start_record=1
                on_record()
        if(start_record==1 and (time.time()-record_time)>0.4):
            if(amplitude>amp):
                start_record=2
                timer1.start()
                
    else:#发送者
        idx = int(2500 / freq_res)
        amplitude = int(np.abs(fft_data[idx]) / 10240)
        ui.textEdit_4.append("当前声音："+str(amplitude))
        if(re_start_record==0):
            if(amplitude>amp):
                
                record_time=time.time()
                re_start_record=1
                on_record()
        if(re_start_record==1 and (time.time()-record_time)>0.4):
            if(amplitude>amp):
                re_start_record=2
                timer1.start()
                

continue_flag=0
continue_count=0
right_count=0

def continue_send():
    #连续发送按钮
    global continue_flag
    continue_flag=1
    continue_send_handle()

def continue_send_handle():
    #连续发送的执行过程
    global calculation,have_code_flag,continue_flag,continue_count,right_count
    if(ui.checkBox.isChecked()==False):
        n=ui.spinBox.value()
        if(continue_count<n):
            calculation=generate_expression()
            have_code_flag=1
            hamming_encode()
            send()
            
            continue_count+=1
        else:
            continue_flag=0
            continue_count=0
            right_count=10
            ui.lineEdit_2.setText(str(right_count))
            ui.lineEdit_3.setText(str(int(right_count/n*100))+'%')
            right_count=0

twice=0
def on_timeout():
    #结束录音前先过0.2s确保录音完全
    global start_record,re_start_record,twice
    twice+=1
    if(twice==2):
        on_stop()
        de_hammingcode()
        re_start_record=0
        start_record=0
        twice=0
        timer1.stop()


if __name__ == "__main__":
 
    app = QApplication(sys.argv)   
    
    MainWindow = QMainWindow()    
    MainWindow.setWindowTitle("wareless")
    ui = Ui_computer.Ui_MainWindow()     # ui=GUI的py文件名.类名
    ui.setupUi(MainWindow)
    MainWindow.show()
    
    formatAudio = QAudioFormat()
    formatAudio.setSampleRate(48000)
    formatAudio.setChannelCount(1)
    formatAudio.setSampleSize(16)
    formatAudio.setCodec("audio/pcm")
    formatAudio.setByteOrder(QAudioFormat.LittleEndian)
    formatAudio.setSampleType(QAudioFormat.SignedInt)
    recorder = Recorder(formatAudio)
    
    ui.pushButton_22.clicked.connect(clear)
    ui.pushButton_25.clicked.connect(clear_2)
    ui.pushButton_23.clicked.connect(finish)
    ui.pushButton_17.clicked.connect(hamming_encode)
    ui.pushButton_18.clicked.connect(send)
    ui.pushButton_19.clicked.connect(on_record)
    ui.pushButton_24.clicked.connect(on_stop)
    ui.pushButton_20.clicked.connect(de_hammingcode)
    ui.pushButton_26.clicked.connect(continue_send)
    timer = QTimer()
    timer.setInterval(40)
    timer.timeout.connect(detect)
    timer.start()
    # 创建定时器
    timer1 = QTimer()
    timer1.setInterval(200)
    timer1.timeout.connect(on_timeout)
    sys.exit(app.exec_())


