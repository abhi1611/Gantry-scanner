import time
import serial
from localization3 import mycapture, ini, Inference
import matplotlib.pyplot as plt
import skimage.transform
from datetime import datetime
arduino = serial.Serial('COM4', 9600)
inc_angle = 3

def visualization():
    img = mycapture()
    area = (500, 100, 1500, 640)  # following x,y
    image = img.crop(area)

    overlay, tensor2,current_prob= Inference(image, model_ft)
    ################

    newsize = (224, 224)
    tensor = image.resize(newsize)
    t0 = time.time()
    plt.close() #Makes it more optimized. Otherwise, it would take incrementally more time to plot

    plt.subplot(1, 2, 1), plt.imshow(tensor)
    plt.subplot(1, 2, 2), plt.imshow(tensor)
    plt.subplot(1, 2, 2), plt.imshow(skimage.transform.resize(overlay[0], tensor2.shape[1:3]), alpha=0.5, cmap='jet')

    t1 = time.time()

    plt.pause(0.005)
    plt.draw()

    #to save cam results
    path1 = "cam_results/"
    fname = path1 + datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p") + ".png"
    plt.savefig(fname)

    print("file saved with name =\t", fname)
    print("iter time:", round(t1 - t0, 3), round(time.time() - t1, 3))
    return current_prob

def gimbal_sweep(pr_steps, step_number, steps):
    dir = step_number%2
    max_roll = 0
    max_pitch = 0

    p_max = 0
    for i in range(-pr_steps, pr_steps+1) if dir == 1 else range(pr_steps, -pr_steps-1, -1):
        # goes through all possible values of the angles and sends them to the arduino
        for j in range(-pr_steps, pr_steps+1) if ((i%2==0 and dir==1) or (i%2==1 and dir==0)) else range(pr_steps, -pr_steps-1, -1):
            print()
            print("Linear step number")
            print(step_number-steps+1)
            print("Angle position")
            print(inc_angle*i, inc_angle*j)
            arduino.write((str(inc_angle*i) + '\n').encode()) #sends the angle values to the arduino for setting it in that position
            while(arduino.inWaiting() == 0):
                pass
            msg = arduino.readline()
            msg = str(msg, 'utf-8').strip('\r\n')

            if(msg == "roll ready"):
                pass
            else:
                print("Error: roll ready not recieved")
            arduino.write((str(inc_angle*j) +'\n').encode())

            while(arduino.inWaiting() == 0):
                pass

            msg = arduino.readline()
            msg = str(msg, 'utf-8').strip('\r\n')

            if(msg == "gimbal_paused"): #only when the pause signal is reccieved from the arduino, it continues
                pass
            else:
                print("Error: gimbal paused not recieved")
                return

            prob = visualization() #visualization function is carried out
            if prob > p_max:
                p_max = prob
                max_pitch = inc_angle*i
                max_roll = inc_angle*j

    
    arduino.write("gimbal_stop\n".encode()) #signal sent to arduino to stop the current gimbal sweep

    return(p_max, max_pitch, max_roll)

steps = int(input("Enter number of steps: "))
pr_steps = int(input("Enter number of gimbal steps on each side: ")) #5 on each side of roll and pitch

model_ft = ini()
for step_count in [200, 40, 8]:    # for steps=5
# for step_count in [200, 20, 2]:  #iterates through denominator of speed   # for steps =10
    arduino.write("linear_sweep\n".encode()) #sends signal to arduino for starting linear sweep
    arduino.write((str(step_count) + '\n').encode()) #sends the denominator for speed to the arduino for the current linear sweep

    for i in range(steps):
        arduino.write("linear_move\n".encode())
        arduino.write("F\n".encode())
        if(i == steps-1):
            arduino.write("gimbal\n".encode())
        else:
            arduino.write("no gimbal\n".encode())

        while (arduino.inWaiting() == 0):
            pass
        msg = arduino.readline()
        msg = str(msg, 'utf-8').strip('\r\n')
        if (msg == "linear_paused"):  # continues with the execution when it has got the stop signal from the arduino
            pass
        else:
            print("Error: linear paused not received")
            break

    p_max, max_pitch, max_roll = gimbal_sweep(pr_steps,-1, steps)
    max_lin = 0

    #forward motion
    for i in range(2*steps): #does one sweeping step in every iteration for the given number of steps
        arduino.write("linear_move\n".encode()) #sends signal to arduino to start current motion for 100 steps
        arduino.write("B\n".encode())
        arduino.write("gimbal\n".encode())

        while(arduino.inWaiting() == 0):
            pass
        msg = arduino.readline()
        msg = str(msg, 'utf-8').strip('\r\n')
        if(msg == 'linear_paused'): #continues with the execution when it has got the stop signal from the arduino
            pass
        else:
            print("Error: linear paused not recieved")
            break
        
        prob, pitch_pos, roll_pos = gimbal_sweep(pr_steps, i, steps)
        if prob > p_max:
            p_max = prob
            max_pitch, max_roll = pitch_pos, roll_pos
            max_lin = i


    

    # #does the same as the previous block but for backward motion
    # for i in range(steps):
    #     arduino.write("linear_move\n".encode())
    #     arduino.write("F\n".encode())
    #     arduino.write("gimbal\n".encode())
    #
    #     while(arduino.inWaiting() == 0):
    #         pass
    #     msg = arduino.readline()
    #     msg = str(msg, 'utf-8').strip('\r\n')
    #     if (msg == 'linear_paused'):
    #         pass
    #     else:
    #         print("Error: linear paused not recieved")
    #         break
    #
    #     gimbal_sweep(pr_steps, scanlist, gimblist)

    arduino.write("linear_stop\n".encode()) #signal sent to arduino to stop linear sweep

    linear_idx = 2*steps - max_lin
    arduino.write((str(linear_idx) + '\n').encode()) #number of steps to backtrack for the maximum position is sent to the arduino

    while(arduino.inWaiting() == 0):
        pass
    msg = arduino.readline()
    msg = str(msg, 'utf-8').strip('\r\n')
    if (msg == "linear_ready"): #waits for signal that says that the linear sweep to the optimal linear position has happened
        pass
    else:
        print("Error: linear ready not recieved")
        break

    # maximum prob pitch and roll position

    # pitch and roll position are sent to arduino
    arduino.write((str(max_pitch) + '\n').encode())
    arduino.write((str(max_roll) + '\n').encode())

    #steps = 10
    steps = 5
    while(arduino.inWaiting() == 0):
        pass
    msg = arduino.readline()
    msg = str(msg, 'utf-8').strip('\r\n')
    if (msg == "gimbal_ready"): # waits for signal saying that gimbal has reached optimal position
        pass
    else:
        print("Error: gimbal ready not recieved")
        break

    prob = visualization()
    print()
    print("Final angle: ", max_pitch, max_roll)
    print("Probability: ", prob)

