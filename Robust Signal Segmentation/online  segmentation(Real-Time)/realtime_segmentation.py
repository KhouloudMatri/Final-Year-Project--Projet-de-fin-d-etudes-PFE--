import time
import matplotlib
import matplotlib.pyplot as plt
import socket
import numpy as np

host = '0.0.0.0'  # Listen on all available network interfaces
port = 5000      # Port to listen on, should match the port used by the ESP32 client

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host, port))
server_socket.listen(5)
# print(f'Server listening on {host}:{port}')


# Create a figure and axis
fig, ax = plt.subplots()
line, = ax.plot([], [], color='b')
ax.set_ylim(0, 330)
x = np.zeros(4000)
y = np.zeros(4000)


i = 0
buffer_size = 33
value_buffer = []
# Enable interactive mode
plt.ion()
# Show the plot window without blocking
plt.show(block=False)

# # Sampling frequency
Fs = 1000  # Assuming 1000 Hz sampling rate

# # Constants
w = 150 # Size of window for correlation coefficient

segment_buff = []
# onset = 0
onset = False
offset = False
r = []  
onset_index     = [0] #to check
onset_buffer    = [0]  #sure
offset_index    = [0] # to check
offset_buffer   = [0] #sure 
signal_buffer   = []
tkeo_buffer     = []

n = -1   #number of read samples                                                     
s = 0    # no activity (1 means activity)


def update_plot(i, values, onset, onset_position, offset, offset_position):
    global x, y
    
    for value in values:
        x = np.roll(x, -1)
        y = np.roll(y, -1)
        x[-1] = i
        y[-1] = value
        i += 1
    
    line.set_data(x, y)
    
    ax.set_xlim(left=max(0, i-4000), right=i)  # Adjust the x-axis limits to scroll

    # Use blitting to speed up rendering
    fig.canvas.restore_region(background)
    ax.draw_artist(line)
    
    # Check for onset and plot vertical line if detected
    
    
    onset_line = ax.axvline(x=onset_position, color='green', linestyle='--')
    ax.draw_artist(onset_line)
        # onset = False
    # Check for offset and plot vertical line if detected
    
    offset_line = ax.axvline(x=offset_position, color='red', linestyle='--')
    ax.draw_artist(offset_line)
        # offset = False
        
    fig.canvas.blit(ax.bbox)
    fig.canvas.flush_events()
    
### Set up blitting
fig.canvas.draw()
background = fig.canvas.copy_from_bbox(ax.bbox)
   



while True:
    client_socket, addr = server_socket.accept()
    print(f'Connection from {addr} has been established.')
     
                    
    data_buffer = ""
    while True:
        data = client_socket.recv(4056)
       
        
        data_buffer = data.decode()
        try:
            # Convert to float first, then to an integer to handle cases like "123.0"
            value = str(int(float(data_buffer)))
            
            
            ### process plotting
            live_dataa = int(value)
            live_data = live_dataa 
            value_buffer.append(live_data)
            # converted_data = (live_data)
            
            
            #####segmentation
            element = live_data
            ##store signal
            signal_buffer.append(element) # Store the signal in the buffer
            n += 1
            #offset = 0
            if n >= 2: #len(signal_buffer) >= 3:
                tkeo = signal_buffer[-2]**2 - signal_buffer[-3] * signal_buffer[-1]
                tkeo_buffer.append(tkeo)
                if n > 2*w+2:                # len(tkeo_buffer) >= 2 * w:
                    tkeo_past = np.abs(tkeo_buffer[-2 * w-2:-w-1])
                    tkeo_future = np.abs(tkeo_buffer[-w-1:]) 
                    r.append(np.corrcoef(tkeo_past, tkeo_future)[0, 1]) # corr_coeff
                                            # check for zero crossing
                    if abs(r[-1]) <= 0.01:  # Change in state detected
                        E1 = sum(np.abs(tkeo_buffer[n-w-w:n-w])) # initial.Energy
                        E2 = sum(np.abs(tkeo_buffer[n-w:n])) # current.Energy
                        if E2 > 2 * E1 : # and E1 > E2*0.05: # and s == 0:
                            onset_index.append(n-w)
                            if s==0:
                                onset_buffer.append(n-w)
                                print(f'onset: {onset_buffer[-1]}')
                                s = 1
                                onset = True
                                offset = False
                                segment_buff = []
                               
                                    
                        if E1 > 2 * E2: # and E1 > E2*0.05: # and s == 0:
                            offset_index.append(n-w)
                            if s == 1:
                                if (n-w-onset_buffer[-1]) > Fs/2:
                                    offset_buffer.append(n-w)
                                    s = 0
                                    offset = True
                                    print(f'offset: {offset_buffer[-1]}')  
                                    
                                    
            #saving segments     
            if  onset and not offset :
                #save function csv
                segment_buff.append(element)
            
            #updating the plot by frame of 33samples
            if len(value_buffer) >= buffer_size:
                update_plot(i, value_buffer, onset, onset_buffer[-1], offset, offset_buffer[-1])
                i += buffer_size
                value_buffer = []
            
            
            
              
            
        
        
        
        #when the data received is a string of values
        except ValueError:
            # print(data_buffer+"The string is not a number that can be converted to an integer.")
            data_buffer += "\n"
            # Process data packets split by newline or other delimiter
            while '\n' in data_buffer: 
                packet, data_buffer = data_buffer.split('\n', 1)
                packet = packet.strip()
                if packet:
                    value = str(int(float(packet)))  # Convert to int
                    #print(f"Received number: {value}")
                    
                    live_dataa = int(value)
                    live_data = live_dataa 
                    value_buffer.append(live_data)
                    # converted_data = (live_data) 
                    
                    
                    ####segmentation
                    element = live_data
                    #store signal 
                    signal_buffer.append(element) # Store the signal in the buffer
                    n += 1
                    #offset = 0
                    if n >= 2: #len(signal_buffer) >= 3:
                        tkeo = signal_buffer[-2]**2 - signal_buffer[-3] * signal_buffer[-1]
                        tkeo_buffer.append(tkeo)
                        if n > 2*w+2:                # len(tkeo_buffer) >= 2 * w:
                            tkeo_past = np.abs(tkeo_buffer[-2 * w-2:-w-1])
                            tkeo_future = np.abs(tkeo_buffer[-w-1:]) 
                            r.append(np.corrcoef(tkeo_past, tkeo_future)[0, 1]) # corr_coeff
                                                    # check for zero crossing
                            if abs(r[-1]) <= 0.01:  # Change in state detected
                                # axs[3].axvline(x = (n-w) /Fs, color='green', linestyle='--', label='Vertical line at x=3')
                                E1 = sum(np.abs(tkeo_buffer[n-w-w:n-w])) # initial.Energy
                                E2 = sum(np.abs(tkeo_buffer[n-w:n])) # current.Energy
                                if E2 > 2 * E1 : # and E1 > E2*0.05: # and s == 0:
                                    onset_index.append(n-w)
                                    if s==0:
                                        onset_buffer.append(n-w)
                                        onset= True
                                        # ax.axvline(x = onset_buffer[-1] , color='green', linestyle='--')
                                        print(f'onset: {onset_buffer[-1]}')
                                        s = 1
                                if E1 > 2 * E2: # and E1 > E2*0.05: # and s == 0:
                                    offset_index.append(n-w)
                                    if s == 1:
                                        if (n-w-onset_buffer[-1]) > Fs/2:
                                            offset_buffer.append(n-w)
                                            s = 0
                                            offset = True
                                            onset = False
                                            # ax.axvline(x = offset_buffer[-1] , color='red', linestyle='--')
                                            print(f'offset: {offset_buffer[-1]}')
                                            
                                            
                    #saving segments     
                    if  onset and not offset :
                        #save function csv
                        segment_buff.append(element)
                        
                    #updating the plot by frame of 33samples
                    if len(value_buffer) >= buffer_size:
                        update_plot(i, value_buffer, onset, onset_buffer[-1], offset, offset_buffer[-1])
                        i += buffer_size
                        value_buffer = []
                    
                    
                    
             
                   
                        
                        
                
    client_socket.close()
    print(f'Connection from {addr} closed.')

