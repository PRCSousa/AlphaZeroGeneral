#em script para eu puder abrir uma duas shells e escrever as moves

import socket
import random
import time

client_mode = int(input("1 - AlphaZero\n2 - Human\n"))

if client_mode == 1:
    pass

elif client_mode == 2:
    #get logic from model we choose
    pass

def connect_to_server(host='localhost', port=12345):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    
    response = client_socket.recv(1024).decode()
    print(f"Server ResponseINIT: {response}")
    
    Game = response[-4:]
    print("Playing:", Game)
    
    if "1" in response:
        ag=1
    else:
        ag=2
    first=True

    while True:
        
        if ag == 1 or not first:

            if Game[0] == 'A':
                move = tuple(int(x.strip()) for x in input("\nInput your move: ").split(' '))
                move_str = ''
                time.sleep(1)
                for elem in move:
                    move_str += str(elem)
                client_socket.send(move_str.encode())
                print("Sending move: ",move)
                response = client_socket.recv(1024).decode()

                if response == 'VALID':
                    continue

                while response == 'INVALID':
                    print("Invalid Move\nSend new move: ")
                    move = tuple(int(x.strip()) for x in input("\nInput your move: ").split(' '))
                    move_str = ''
                    time.sleep(1)
                    for elem in move:
                        move_str += str(elem)
                    client_socket.send(move_str.encode())
                    print("Sending move: ",move)

            elif Game[0] == 'G':
                a, b = tuple(int(x.strip()) for x in input("\nInput your move: ").split(' '))
                print("\n")
                move = a * 9 + b

                if response == 'VALID':
                    continue

                while response == 'INVALID':
                    print("Invalid Move\nSend new move: ")
                    a, b = tuple(int(x.strip()) for x in input("\nInput your move: ").split(' '))
                    print("\n")
                    move = a * 9 + b
                    time.sleep(1)
                    client_socket.send(move.encode())
                    print("Sending move: ",move)

            # Wait for server response
            response = client_socket.recv(1024).decode()
            print(f"Server Response1: {response}")
            if "END" in response: break
         
        first=False
        response = client_socket.recv(1024).decode()
        print(f"Server Response2: {response}")
        if "END" in response: break


    client_socket.close()

if __name__ == "__main__":
    connect_to_server()