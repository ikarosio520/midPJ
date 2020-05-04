
#include "mbed.h"
#include <string.h>
// for uLCD
#include "uLCD_4DGL.h"
// for DNN and ACC
#include "accelerometer_handler.h"
#include "config.h"
#include "magic_wand_model_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
// for Audio
#include <cmath>
#include "DA7212.h"
#define bufferLength (32)
#define signalLength (42)
// PC communicate
Serial pc(USBTX, USBRX);
// audio using
DA7212 audio;
int16_t waveform[kAudioTxBufferSize];
char serialInBuffer[bufferLength];
// thread variable
EventQueue queue(32 * EVENTS_EVENT_SIZE);
EventQueue queue2(32 * EVENTS_EVENT_SIZE);
Thread checking(osPriorityNormal, 16 * 1024); // 16K stack size
Thread musicing(osPriorityNormal, 16 * 1024); // 16K stack size
Thread playing (osPriorityNormal, 16 * 1024); // 16K stack size
// func variable
InterruptIn MENU(SW2);
DigitalIn BW(SW3);
DigitalOut red_led(LED1);
DigitalOut green_led(LED2); // light up when load data
DigitalOut y_led(LED3);
int mode_ = 2;
volatile bool playornot = true;
// for DNN setting
constexpr int kTensorArenaSize = 60 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
static tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter *error_reporter = &micro_error_reporter;
const tflite::Model *model = tflite::GetModel(g_magic_wand_model_data);
static tflite::MicroOpResolver<6> micro_op_resolver;
// saved song data
char song_name[3][20] = {"Little Star", "song1", "song2"};
int song_index = 0;
int step = 0;
int song[42] = {
    261, 261, 392, 392, 440, 440, 392,
    349, 349, 330, 330, 294, 294, 261,
    392, 392, 349, 349, 330, 330, 294,
    392, 392, 349, 349, 330, 330, 294,
    261, 261, 392, 392, 440, 440, 392,
    349, 349, 330, 330, 294, 294, 261};

int noteLength[42] = {
    1, 1, 1, 1, 1, 1, 2,
    1, 1, 1, 1, 1, 1, 2,
    1, 1, 1, 1, 1, 1, 2,
    1, 1, 1, 1, 1, 1, 2,
    1, 1, 1, 1, 1, 1, 2,
    1, 1, 1, 1, 1, 1, 2};
// uLCD port
uLCD_4DGL uLCD(D1, D0, D2); // serial tx, serial rx, reset pin;
// play single tone
void playNote(int freq)
{
  for (int i = 0; i < kAudioTxBufferSize; i++)
  {
    waveform[i] = (int16_t)(sin((double)i * 2. * M_PI / (double)(kAudioSampleFrequency / freq)) * ((1 << 16) - 1));
  }
  // the loop below will play the note for the duration of 1s
  for (int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize; ++j)
  {
    audio.spk.play(waveform, kAudioTxBufferSize);
  }
}
// backward the music
void back(void)
{
  if (song_index >0) song_index--;
  else song_index = 2;
}
// forward the music
void forward(void)
{
  if (song_index <2) song_index++;
  else song_index = 0;
}
// load song data into board
void loadSignal(void)
{
  green_led = 0;
  int i = 0;
  int serialCount = 0;
  //audio.apk.plause();
  pc.printf("%d\n", song_index);
  while (i < signalLength)
  {
    if (pc.readable())
    {
      serialInBuffer[serialCount] = pc.getc();
      serialCount++;
      if (serialCount == 3)
      {
        serialInBuffer[serialCount] = '\0';
        song[i] = (float)atof(serialInBuffer);
        serialCount = 0;
        i++;
      }
    }
  }
  i = 0;
  while (i < signalLength)
  {
    if (pc.readable())
    {
      serialInBuffer[serialCount] = pc.getc();
      serialCount++;
      if (serialCount == 1)
      {
        serialInBuffer[serialCount] = '\0';
        noteLength[i] = (float)atof(serialInBuffer);
        serialCount = 0;
        i++;
      }
    }
  }
  green_led = 1;
}
// Return the result of the last prediction
int PredictGesture(float *output)
{
  // How many times the most recent gesture has been matched in a row
  static int continuous_count = 0;
  // The result of the last prediction
  static int last_predict = -1;

  // Find whichever output has a probability > 0.8 (they sum to 1)
  int this_predict = -1;
  for (int i = 0; i < label_num; i++)
  {
    if (output[i] > 0.8)
      this_predict = i;
  }

  // No gesture was detected above the threshold
  if (this_predict == -1)
  {
    continuous_count = 0;
    last_predict = label_num;
    return label_num;
  }

  if (last_predict == this_predict)
  {
    continuous_count += 1;
  }
  else
  {
    continuous_count = 0;
  }
  last_predict = this_predict;

  // If we haven't yet had enough consecutive matches for this gesture,
  // report a negative result
  if (continuous_count < config.consecutiveInferenceThresholds[this_predict])
  {
    return label_num;
  }
  // Otherwise, we've seen a positive result, so clear all our variables
  // and report it
  continuous_count = 0;
  last_predict = -1;

  return this_predict;
}
void CheckG(void)
{
  //initialization DNN
  micro_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
                               tflite::ops::micro::Register_MAX_POOL_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                               tflite::ops::micro::Register_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                               tflite::ops::micro::Register_FULLY_CONNECTED());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                               tflite::ops::micro::Register_SOFTMAX());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
                               tflite::ops::micro::Register_RESHAPE(), 1);
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  tflite::MicroInterpreter *interpreter = &static_interpreter;
  interpreter->AllocateTensors();
  TfLiteTensor *model_input = interpreter->input(0);
  int input_length = model_input->bytes / sizeof(float);
  TfLiteStatus setup_status = SetupAccelerometer(error_reporter);

  TfLiteTensor *model_inputtt = model_input;
  tflite::MicroInterpreter *interpreterrr = interpreter;
  int input_lengthhh = input_length;
  bool should_clear_buffer = false;
  bool got_data = false;
  int gesture_index;
  while (true)
  {
    if (BW == 0)
    {
      break;
    }
    else
    {
      got_data = ReadAccelerometer(model_inputtt->data.f, input_lengthhh, should_clear_buffer);
      if (!got_data)
      {
        should_clear_buffer = false;
        continue;
      }
      TfLiteStatus invoke_status = interpreterrr->Invoke();
      if (invoke_status != kTfLiteOk)
      {
        continue;
      }
      gesture_index = PredictGesture(interpreterrr->output(0)->data.f);
      should_clear_buffer = gesture_index < label_num;
      if (gesture_index == 1)
      {
        if (mode_ == 0)
        {
          uLCD.printf("\nselecting: forward\n");
        }
        if (mode_ == 1)
        {
          uLCD.printf("\nselecting: change song\n");
        }
        if (mode_ == 2)
        {
          uLCD.printf("\nselecting: backward\n");
        }
        if (mode_ < 2)
          mode_++;
        else
          mode_ = 0;
      }
    }
  }
}
void playinging(void)
{
  while (1)
  {

    for (; step < 42; step++)
    {
      if (playornot == true)
      {
        uLCD.cls();
        uLCD.printf("\nPlaying~~\n");
        uLCD.printf("\n%2D %s %2D\n", song[step], song_name[song_index], step);
        int length = noteLength[step];
        while (length--)
        {
          queue2.call(playNote, song[step]);
          if (length <= 1)
            wait(1.0);
        }
      }
      else
      {
        wait(1);
      }
    }
    step = 0;
  }
}
// selecting MENU
void mode_select(void)
{
  // stop playing
  playornot = false;
  green_led = 1;
  red_led = 0;
  y_led = 1;
  uLCD.cls();
  wait(1.5);
  uLCD.printf("\nMODE menu:\n");
  while (true)
  {
    wait(1);
    CheckG();
    green_led = 1;
    red_led = 1;
    y_led = 1;
    uLCD.cls();
    if (mode_ == 1)
    {
      uLCD.printf("\nselect!!! forward\n");
      forward();
      uLCD.printf("\nchoose_index; %2D\n", song_index);
      step = 0;
      loadSignal();
      wait(1);
      break;
    }
    if (mode_ == 2)
    {
      uLCD.printf("\nselect!!! change song\n");
      while (1)
      {
        if (BW == 0)
        {
          if (song_index < 2)
            song_index++;
          else
            song_index = 0;
          uLCD.cls();
          uLCD.printf("\nsong_index; %2D\n", song_index);
          wait(2);
        }
        else
        {
          uLCD.printf("\nchoose_index; %2D\n", song_index);
          step = 0;
          loadSignal();
          wait(1);
          break;
        }
      }
    }
    if (mode_ == 0)
    {
      uLCD.printf("\nselect!!! backward\n");
      back();
      uLCD.printf("\nchoose_index; %2D\n", song_index);
      step = 0;
      loadSignal();
      wait(1);
      break;
    }
    wait(1);
    playornot = true;
    break;
  }
}
int main(void)
{
  //initialization setting
  pc.baud(9600);
  green_led = 1;
  red_led = 1;
  y_led = 1;
  // Running
  uLCD.printf("\n106061118\n");
  wait(2);
  // MENU button event trigger
  checking.start(callback(&queue, &EventQueue::dispatch_forever));
  MENU.fall(queue.event(mode_select));
  //
  playing.start(callback(&queue2, &EventQueue::dispatch_forever));
  // PlayMusic thread going at background
  musicing.start(playinging);
}
