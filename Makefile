all: lane_detection_seq lane_detection_pthread lane_detection_openmp_v1 collect

lane_detection_seq:
	g++ -Wall -g -std=c++1y -O3 lane_detection_seq.cpp -o run_lane_detection_seq -lpthread `pkg-config --cflags --libs opencv`

lane_detection_pthread:
	g++ -Wall -g -std=c++1y -O3 lane_detection_pthread.cpp -o run_lane_detection_pthread -lpthread `pkg-config --cflags --libs opencv`

lane_detection_openmp_v1:
	g++ -Wall -g -std=c++1y -O3 lane_detection_openmp_v1.cpp -o run_lane_detection_openmp_v1 -fopenmp `pkg-config --cflags --libs opencv`

collect:
	gcc collect.c -o collect

clean:
	rm -rf logs -R
	rm -rf run_lane_detection_seq
	rm -rf result_seq.avi
	rm -rf run_lane_detection_pthread
	rm -rf result_pthread.avi
	rm -rf run_lane_detection_openmp_v1
	rm -rf result_openmp_v1.avi
	rm -rf collect
