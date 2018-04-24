/**
 * ------------------------------------------------------------------------------------------
 * Lane Detection:
 *
 * General idea and some code modified from:
 * chapter 7 of Computer Vision Programming using the OpenCV Library. 
 * by Robert Laganiere, Packt Publishing, 2011.
 * This program is free software; permission is hereby granted to use, copy, modify, 
 * and distribute this source code, or portions thereof, for any purpose, without fee, 
 * subject to the restriction that the copyright notice may not be removed 
 * or altered from any source or altered source distribution. 
 * The software is released on an as-is basis and without any warranties of any kind. 
 * In particular, the software is not guaranteed to be fault-tolerant or free from failure. 
 * The author disclaims all warranties with regard to this software, any use, 
 * and any consequent failure, is purely the responsibility of the user.
 *
 * Copyright (C) 2013 Jason Dorweiler, www.transistor.io
 * ------------------------------------------------------------------------------------------
 * Source:
 *
 * http://www.transistor.io/revisiting-lane-detection-using-opencv.html
 * https://github.com/jdorweiler/lane-detection
 * ------------------------------------------------------------------------------------------
 * Notes:
 * 
 * Add up number on lines that are found within a threshold of a given rho,theta and 
 * use that to determine a score.  Only lines with a good enough score are kept. 
 *
 * Calculation for the distance of the car from the center.  This should also determine
 * if the road in turning.  We might not want to be in the center of the road for a turn. 
 *
 * Several other parameters can be played with: min vote on houghp, line distance and gap.  Some
 * type of feed back loop might be good to self tune these parameters. 
 * 
 * We are still finding the Road, i.e. both left and right lanes.  we Need to set it up to find the
 * yellow divider line in the middle. 
 * 
 * Added filter on theta angle to reduce horizontal and vertical lines. 
 * 
 * Added image ROI to reduce false lines from things like trees/powerlines
 * ------------------------------------------------------------------------------------------
 */

/**
 * ------------------------------------------------------------------------------------------
 * Versão paralela do algoritmo Lane-Dectection com OpenMP.
 * 
 * Autor: Gabriell A. de Araujo (hexenoften@gmail.com)
 *
 * Última modificação: (24/04/2018)
 *
 * Copyright (C) 2018 Gabriell A. de Araujo, GMAP-PUCRS (http://www.inf.pucrs.br/gmap)
 * ------------------------------------------------------------------------------------------
 * Comando de compilação:
 *
 * g++ -Wall -g -std=c++1y -O3 lane_detection_openmp.cpp -o run_lane_detection_openmp -fopenmp `pkg-config --cflags --libs opencv`
 * ------------------------------------------------------------------------------------------
 * Comando de execução:
 *
 * export OMP_WAIT_POLICY=PASSIVE && ./run_lane_detection_openmp <video_file_name> <number_of_threads>
 * ------------------------------------------------------------------------------------------
 * Notas:
 *
 * O laço while da stream é executado de forma concorrente por todas as threads;
 *
 * A cada iteração, cada thread pega um frame do vídeo, processa este frame e o envia para a fila de saída;
 *
 * A cada iteração, uma única thread verifica a fila de saída e caso o frame esperado esteja lá, o envia para o display;
 *
 * A estrutura da fila é necessária para que os frames possam ser enviados de maneira ordenada para o display.
 * ------------------------------------------------------------------------------------------
 * Comandos de instalação do opencv:
 *
 * sudo apt-get install gcc libopencv-dev
 * sudo apt-get install g++ libopencv-dev
 * ------------------------------------------------------------------------------------------
 */

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <stdbool.h>
#include <stdio.h>
#include <omp.h>
#include "linefinder.h"

using namespace cv;
using namespace std;

//estruturas
typedef struct node 
{	
	cv::Mat* frame;
	int frame_number;
	bool is_the_last_node;
	struct node* next;
} work_node;

//variáveis globais
bool is_there_any_frame;
bool is_there_any_work;
bool finished;
VideoWriter oVideoWriter;
VideoCapture capture;
int nframes;
int threads_number;
work_node* input_head;
work_node* input_tail;
work_node* output_head;
work_node* output_tail;
int size_input_work_queue;
int size_output_work_queue;
int current_frame;
omp_lock_t input_work_queue_lock;
omp_lock_t output_work_queue_lock;
omp_lock_t get_frame_lock;
int correctness; 

//protótipos de funções
int get_head_id_from_input_work_queue();
int get_head_id_from_output_work_queue();
void add_to_input_work_queue(work_node** node_aux);
void add_to_output_work_queue(work_node** node_aux);
work_node* remove_from_input_work_queue();
work_node* remove_from_output_work_queue();
void process_frame(work_node** node_aux);
void send_frame_to_display(work_node** node_aux);
void parallel_processing();

/**
 * Pré-condições:
 * -Deve existir pelo menos um nodo na fila encadeada de entrada.
 *
 * Pós-condições:
 * -Retorna o id do primeiro nodo de trabalho da fila encadeada de entrada.
 */
int get_head_id_from_input_work_queue()
{
	return input_head->frame_number; 
} 

/**
 * Pré-condições:
 * -Deve existir pelo menos um nodo na fila encadeada de saída.
 *
 * Pós-condições:
 * Retorna o id do primeiro nodo de trabalho da fila encadeada de saída.
 */
int get_head_id_from_output_work_queue()
{
	return output_head->frame_number; 
} 

/**
 * Pré-condições:
 * -O argumento da função não pode ser nulo.
 *
 * Pós-condições:
 * -Adiciona um nodo de trabalho na última posição da fila encadeada de entrada.
 * -Incrementa o valor da variável size_input_work_queue.
 */
void add_to_input_work_queue(work_node** node_aux)
{
	//adiciona o nodo na primeira posição se a fila está vazia
	if(size_input_work_queue == 0)
	{
		input_head = *node_aux;

		input_tail = input_head;

		size_input_work_queue++;

		return;			
	}
	//adiciona o nodo na última posição se a fila não está vazia
	else
	{
		input_tail->next = *node_aux;

		input_tail = input_tail->next;

		size_input_work_queue++;

		return;
	}	
}

/**
 * Pré-condições:
 * -O argumento da função não pode ser nulo.
 *
 * Pós-condições: 
 * -Adiciona um nodo de trabalho ordenado pelo id do frame na fila encadeada de saída.
 * -Incrementa o valor da variável size_output_work_queue.
 */
void add_to_output_work_queue(work_node** node_aux)
{
	//adiciona o nodo na primeira posição se a fila está vazia
	if(size_output_work_queue == 0)
	{
		output_head = *node_aux;

		output_tail = output_head;

		size_output_work_queue++;

		return;			
	}
	else
	{		
		int node_aux_id = (*node_aux)->frame_number;	

		work_node* current_node = output_head;
		int current_id = output_head->frame_number;

		//adiciona o nodo na primeira posição se o id do frame é menor que o da primeira posição
		if(node_aux_id <= current_id)
		{
			(*node_aux)->next = current_node;

			output_head = (*node_aux);

			size_output_work_queue++;

			return;		
		}
		else
		{
			//percorre a fila procurando um nodo de id maior
			while((current_node->next) != NULL)
			{
				current_id = (*(current_node->next)).frame_number;

				//adiciona o nodo na posição imediatamente anterior ao encontrar um nodo de id maior
				if(node_aux_id <= current_id)
				{
					(*node_aux)->next = current_node->next;
					current_node->next = (*node_aux);

					size_output_work_queue++;

					return;
				}
				else
				{
					current_node = current_node->next;
				}
			}

			//adiciona o nodo no final da fila, caso não tenha encontrado um nodo de id maior
			(*node_aux)->next = current_node->next;
			current_node->next = (*node_aux);

			output_tail = (*node_aux);

			size_output_work_queue++;

			return;
		}	
	}
} 

/**
 * Pré-condições:
 * -Deve existir pelo menos um nodo na fila encadeada de entrada.
 *
 * Pós-condições: 
 * -Remove o primeiro nodo da fila encadeada de entrada e retorna o seu endereço.
 * -Decrementa o valor da variável size_input_work_queue.
 * -O next do nodo removido aponta para NULL.
 */
work_node* remove_from_input_work_queue()
{
	work_node* node_to_remove = input_head;

	input_head = input_head->next;

	node_to_remove->next = NULL;

	size_input_work_queue--;	

	return node_to_remove;
} 

/**
 * Pré-condições:
 * -Deve existir pelo menos um nodo na fila encadeada de saída.
 *
 * Pós-condições: 
 * -Remove o primeiro nodo da fila encadeada de saída e retorna o seu endereço.
 * -Decrementa o valor da variável size_output_work_queue.
 * -O next do nodo removido aponta para NULL.
 */
work_node* remove_from_output_work_queue()
{
	work_node* node_to_remove = output_head;

	output_head = output_head->next;

	node_to_remove->next = NULL;

	size_output_work_queue--;	

	return node_to_remove;
}

/**
 * Pré-condições:
 * -O nodo recebido não pode ser nulo.
 *
 * Pós-condições: 
 * -Processa o frame do nodo de trabalho recebido;
 * -Armazena armazena o frame processado no próprio nodo de trabalho recebido.
 */
void process_frame(work_node** node_aux)
{
	//copia o frame do nodo recebido para que o processo seja efetuado
	cv::Mat image = *((*node_aux)->frame);

	int houghVote = 200;
	Mat gray;
	cvtColor(image,gray,CV_RGB2GRAY);
	vector<string> codes;
	Mat corners;
	findDataMatrix(gray, codes, corners);
	drawDataMatrixCodes(image, codes, corners);

	Rect roi(0,image.cols/3,image.cols-1,image.rows - image.cols/3);
	Mat imgROI = image(roi);

	//canny algorithm
	Mat contours;
	Canny(imgROI,contours,50,250);
	Mat contoursInv;
	threshold(contours,contoursInv,128,255,THRESH_BINARY_INV);

	vector<Vec2f> lines;

	if (houghVote < 1 || lines.size() > 2) 
	{ 
		//we lost all lines. reset 
		houghVote = 200; 
	}
	else
	{ 
		houghVote += 25;
	} 

	while(lines.size() < 5 && houghVote > 0)
	{
		HoughLines(contours,lines,1,PI/180, houghVote);
		houghVote -= 5;  
	}

	Mat result(imgROI.size(),CV_8U,Scalar(255));
	imgROI.copyTo(result);

	//draw the limes
	vector<Vec2f>::const_iterator it;
	Mat hough(imgROI.size(),CV_8U,Scalar(0));
	it = lines.begin();

	while(it!=lines.end()) 
	{
		//first element is distance rho
		float rho= (*it)[0]; 
		//second element is angle theta	  
		float theta= (*it)[1]; 			
		if ( (theta > 0.09 && theta < 1.48) || (theta < 3.14 && theta > 1.66) ) 
		{ 	
			//filter to remove vertical and horizontal lines
			//point of intersection of the line with first row
			Point pt1(rho/cos(theta),0);
			//point of intersection of the line with last row
			Point pt2((rho-result.rows*sin(theta))/cos(theta),result.rows);
			//draw a white line
			line(result, pt1, pt2, Scalar(255), 8); 
			line(hough, pt1, pt2, Scalar(255), 8);
		}
		++it;
	}

	//create LineFinder instance			
	LineFinder ld;
	// Set probabilistic Hough parameters
	ld.setLineLengthAndGap(60,10);
	ld.setMinVote(4);

	//detect lines
	vector<Vec4i> li= ld.findLines(contours);
	Mat houghP(imgROI.size(),CV_8U,Scalar(0));
	ld.setShift(0);
	ld.drawDetectedLines(houghP);

	//bitwise AND of the two hough images
	bitwise_and(houghP,hough,houghP);
	Mat houghPinv(imgROI.size(),CV_8U,Scalar(0));
	//threshold and invert to black lines
	threshold(houghP,houghPinv,150,255,THRESH_BINARY_INV); 

	Canny(houghPinv,contours,100,350);
	li= ld.findLines(contours);

	//set probabilistic hough parameters
	ld.setLineLengthAndGap(5,2);
	ld.setMinVote(1);
	ld.setShift(image.cols/3);
	ld.drawDetectedLines(image);

	stringstream stream;
	stream << "Line Segments: " << lines.size();

	putText(image, stream.str(), Point(10,image.rows-10), 2, 0.8, Scalar(0,0,255),0);

	lines.clear();

	//armazena o frame processado no nodo de trabalho recebido
	*((*node_aux)->frame) = image.clone();
} 

/**
 * Pré-condições:
 * -O nodo recebido não pode ser nulo.
 *
 * Pós-condições: 
 * -Envia o frame do nodo de trabalho para o display e desaloca o nodo.
 */
void send_frame_to_display(work_node** node_aux)
{
	cv::Mat* frame_aux = (*node_aux)->frame;

	oVideoWriter.write(*frame_aux);	

	//printf("[[[[THE FRAME id=%d HAS SENT TO THE DISPLAY!!!]]]]\n", (*node_aux)->frame_number);

	//correctness calculation
	correctness = correctness + (*node_aux)->frame_number;

	free(frame_aux);
	free(*node_aux);
}  

/**
 * Pós-condições: 
 * Executa o algoritmo usando processamento paralelo.
 */
void parallel_processing() 
{
	is_there_any_frame = true;
	is_there_any_work = true;
	finished = false;

	omp_set_num_threads(threads_number+1);

	omp_init_lock(&input_work_queue_lock);
	omp_init_lock(&output_work_queue_lock);
	omp_init_lock(&get_frame_lock);

	#pragma omp parallel shared(is_there_any_frame, is_there_any_work, finished)	
	{
		while(finished == false)
		{
			work_node* node_aux;	
			int id_of_my_frame;	
			Mat image;

			///////////////////////
			//####//BLOCK ONE//####
			///////////////////////
			#pragma omp master
			{
				//se ainda existem frames para serem capturados
				if(is_there_any_frame == true)
				{
					capture >> image;

					//se o frame capturado é vazio, envia nodo falso que sinaliza o término da stream
					if (image.empty()) 
					{
						id_of_my_frame = nframes;

						//atualiza o id que será utilizado pelo próximo frame
						nframes++;

						//prepara o nodo de trabalho
						node_aux =  (work_node*) malloc(sizeof(work_node));	
						node_aux->frame = NULL;	
						node_aux->frame_number = id_of_my_frame;	
						node_aux->next = NULL;			
						node_aux->is_the_last_node = true;

						//####//ATIVA LOCK DA FILA DE SAIDA//####//
						omp_set_lock(&input_work_queue_lock);

						//envia o nodo de trabalho para a fila de entrada
						add_to_input_work_queue(&node_aux);

						//####//DESATIVA LOCK DA FILA DE SAIDA//####//
						omp_unset_lock(&input_work_queue_lock);

						is_there_any_frame = false;
					}
					//processa o frame e o adiciona na fila de saída
					else
					{
						id_of_my_frame = nframes;

						//atualiza o id que será utilizado pelo próximo frame
						nframes++;

						//prepara o nodo de trabalho
						node_aux =  (work_node*) malloc(sizeof(work_node));		
						node_aux->frame = new cv::Mat;
						(*(node_aux->frame)) = image.clone();			
						node_aux->frame_number = id_of_my_frame;	
						node_aux->next = NULL;
						node_aux->is_the_last_node = false;

						//####//ATIVA LOCK DA FILA DE SAIDA//####//
						omp_set_lock(&input_work_queue_lock);

						//envia o nodo de trabalho para a fila de entrada
						add_to_input_work_queue(&node_aux);

						//####//DESATIVA LOCK DA FILA DE SAIDA//####//
						omp_unset_lock(&input_work_queue_lock);
					}
				}
			}
			///////////////////////
			//####//BLOCK TWO//####
			///////////////////////
			if( (omp_get_thread_num() != 0) && (is_there_any_work == true) )
			{
				//####//ATIVA LOCK DA FILA DE ENTRADA//####//
				omp_set_lock(&input_work_queue_lock);

				if(size_input_work_queue > 0)
				{
					//remove o nodo da fila de entrada
					node_aux = remove_from_input_work_queue();

					//####//DESATIVA LOCK DA FILA DE ENTRADA//####//
					omp_unset_lock(&input_work_queue_lock);

					//se este é um dos nodos que sinaliza o término da stream, envia o nodo para o próximo estágio e encerra a execução
					if((node_aux->is_the_last_node) == true)
					{			
						//####//ATIVA LOCK DA FILA DE SAIDA//####//
						omp_set_lock(&output_work_queue_lock);

						//envia nodo para a fila do terceiro estágio
						add_to_output_work_queue(&node_aux);

						//####//DESATIVA LOCK DA FILA DE SAIDA//####//
						omp_unset_lock(&output_work_queue_lock);

						is_there_any_work = false;			
					}
					//envia um nodo de trabalho padrão para o processamento do terceiro estágio
					else
					{
						//####//PROCESSA FRAME//####//
						process_frame(&node_aux);

						//####//ATIVA LOCK DA FILA DE SAIDA//####//
						omp_set_lock(&output_work_queue_lock);

						//envia nodo para fila do terceiro estágio
						add_to_output_work_queue(&node_aux);

						//####//DESATIVA LOCK DA FILA DE SAIDA//####//
						omp_unset_lock(&output_work_queue_lock);
					}
				}
				else
				{
					//####//DESATIVA LOCK DA FILA DE ENTRADA//####//
					omp_unset_lock(&input_work_queue_lock);
				}
			}
			/////////////////////////
			//####//BLOCK THREE//####
			/////////////////////////
			#pragma omp master
			{			
				//####//ATIVA LOCK DA FILA DE SAIDA//####//
				omp_set_lock(&output_work_queue_lock);

				//verifica se existe pelo menos um frame na fila
				if(size_output_work_queue > 0)
				{
					int id_of_the_head_of_the_queue = get_head_id_from_output_work_queue();

					//verifica se este frame é o correto a ser enviado para o display
					if(id_of_the_head_of_the_queue == current_frame)
					{
						//remove nodo da fila de saída
						node_aux = remove_from_output_work_queue();

						//####//DESATIVA LOCK DA FILA DE SAIDA//####//
						omp_unset_lock(&output_work_queue_lock);

						//se é o nodo que sinaliza o final da stream, o algoritmo deve encerrar
						if((node_aux->is_the_last_node) == true)
						{
							finished = true;	
						}
						else
						{						
							//manda o frame para o display
							send_frame_to_display(&node_aux);						

							//atualiza o id do frame esperado 
							current_frame++;
						}
					}	
					else
					{
						//####//DESATIVA LOCK DA FILA DE SAIDA//####//
						omp_unset_lock(&output_work_queue_lock);			
					}	
				}
				else
				{
					//####//DESATIVA LOCK DA FILA DE SAIDA//####//
					omp_unset_lock(&output_work_queue_lock);
				}
			}		
		}
	}

	//desaloca os nodos falsos restantes
	while(size_output_work_queue > 0)
	{
		free(remove_from_output_work_queue());
	}	
} 

int main(int argc, char* argv[]) 
{
	string arg = argv[1];
	threads_number = atoi(argv[2]);

	input_head = NULL;
	input_tail = NULL;
	output_head = NULL;
	output_tail = NULL;
	size_input_work_queue = 0;
	size_output_work_queue = 0;
	current_frame = 1;
	nframes = 1;
	correctness = 0;

	//disabling internal OpenCV's support for multithreading. Necessary for more clear performance comparison.
	setNumThreads(0); 

	capture.open(arg);

	//if this fails, try to open as a video camera, through the use of an integer param
	if (!capture.isOpened()) 
	{capture.open(atoi(arg.c_str()));}

	//get the width of frames of the video
	double dWidth = capture.get(CV_CAP_PROP_FRAME_WIDTH); 
	//get the height of frames of the video
	double dHeight = capture.get(CV_CAP_PROP_FRAME_HEIGHT); 

	Size frameSize(static_cast<int>(dWidth), static_cast<int>(dHeight));
	//initialize the VideoWriter object 
	oVideoWriter.open("result_openmp_v2.avi", CV_FOURCC('P','I','M','1'), 20, frameSize, true); 	

	auto tstart = std::chrono::high_resolution_clock::now();

	parallel_processing();	

	auto tend = std::chrono::high_resolution_clock::now();

	double TT; //TIME
	TT = std::chrono::duration<double>(tend-tstart).count();
	double TR = nframes/TT; //FRAMES

	cout << "EXECUTION_TIME_IN_SECONDS: " << TT << endl;
	cout << "FRAMES_PER_SECOND: " << TR << endl;
	printf("CORRECTNESS: %d\n\n", correctness);

	return 0;
}
