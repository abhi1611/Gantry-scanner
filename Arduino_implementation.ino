#include <Servo.h>
#include <AFMotor.h>

Servo pitch_motor;
Servo roll_motor;

int initial_step_speed = 80;
int initial_step_count = 200;

AF_Stepper step_motor(200,1);

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  pitch_motor.attach(10);
  roll_motor.attach(9);

  step_motor.setSpeed(initial_step_speed);
}

void loop() {
  // put your main code here, to run repeatedly:
  while(Serial.available() == 0){
  }
  int step_count;
  if(Serial.readStringUntil('\n') == "linear_sweep"){
    while(Serial.available() == 0){
    }
    step_count = Serial.readStringUntil('\n').toInt();
  }

  step_motor.setSpeed(initial_step_speed);

  while(true){
    while(Serial.available() == 0){
    }
    String msg = Serial.readStringUntil('\n');
    if(msg == "linear_move"){
      while(Serial.available() == 0){}
      String direction =Serial.readStringUntil('\n');
      String gimbal_sweep_state = Serial.readStringUntil('\n');

      if(direction == "F"){
        step_motor.step(step_count, FORWARD, MICROSTEP);
      }
      else if(direction == "B"){
        step_motor.step(step_count, BACKWARD, MICROSTEP);
      }
      delay(1000);
      Serial.println("linear_paused");

      if(gimbal_sweep_state == "gimbal"){
        while(true){
          while(Serial.available() == 0){}
          String msg = Serial.readStringUntil('\n');
          
          if(msg == "gimbal_stop"){
            break;
          }
          
          int pitch_angle = 90 + msg.toInt();
          Serial.println("roll ready");
          while(Serial.available()==0){}
          int roll_angle = 90 + Serial.readStringUntil('\n').toInt();

          pitch_motor.write(pitch_angle);
          roll_motor.write(roll_angle);
          delay(500);
          Serial.println("gimbal_paused");
        }
      }
      else if(gimbal_sweep_state == "no gimbal"){}
      
    }
    else if(msg == "linear_stop"){
      break;
    }
    
  }

  while(Serial.available() == 0){}

  int opt_pos = Serial.readStringUntil('\n').toInt();
  
  for(int i = 0; i < opt_pos; i++){
    step_motor.step(step_count, FORWARD, MICROSTEP);
    delay(1000);
  }
  Serial.println("linear_ready");

  while(Serial.available() == 0){}

  int pitch_pos = 90 + Serial.readStringUntil('\n').toInt();
  int roll_pos = 90 + Serial.readStringUntil('\n').toInt();

  pitch_motor.write(pitch_pos);
  roll_motor.write(roll_pos);

  delay(500);
  Serial.println("gimbal_ready");
  


  
}
