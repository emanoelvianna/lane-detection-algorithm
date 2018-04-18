all: serial	

serial: LaneDetect.cpp
	g++ -std=c++1y -O3 serial_LaneDetect.cpp -o run_serial `pkg-config --cflags --libs opencv`

clean:
	rm -r run_serial
