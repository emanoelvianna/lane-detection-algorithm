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
 * Versão paralela do algoritmo Lane-Dectection com POSIX Threads.
 * 
 * Autor: Gabriell A. de Araujo (hexenoften@gmail.com)
 *
 * Última modificação: (23/02/2018)
 *
 * Copyright (C) 2018 Gabriell A. de Araujo, GMAP-PUCRS (http://www.inf.pucrs.br/gmap)
 * ------------------------------------------------------------------------------------------
 * Comando de compilação:
 *
 * g++ -Wall -g -std=c++1y -O3 lane_detection_pthread.cpp -o run_lane_detection_pthread -lpthread `pkg-config --cflags --libs opencv`
 * ------------------------------------------------------------------------------------------
 * Comando de execução:
 *
 * ./run_lane_detection_pthread <video_file_name> <number_of_threads>
 * ------------------------------------------------------------------------------------------
 * Notas:
 *
 * O algoritmo utiliza a técnica producer-consumer work queue para dividir o trabalho entre as threads;
 * 
 * O algoritmo é divido em 3 estágios;
 * 
 * No primeiro estágio (executado por uma thread), os frames são lidos e mandados para a fila de entrada;
 * 
 * No segundo estágio (executado por n threads), os frames são retirados da fila de entrada, processados e 
 * então enviados para a fila de saída;
 *
 * No terceiro estágio (executado por uma thread), os frames são retirados da fila de saída e enviados para o display;
 * 
 * A implementação faz uso de estruturas encadeadas (filas encadeadas) para otimizar o custo computacional das diversas
 * funções do algoritmo, bem como otimizar o uso de memória;
 * 
 * A função de adicionar nodos na fila de saída, os insere ordenados pelo seu id, de forma que não se torna necessário
 * ordenar os frames para enviá-los para o display;
 *
 * A variável booleana is_the_last_node na estrutura do nodo de trabalho, serve para informar o término do stream
 * de frames. 
 * 
 * Quando a thread do primeiro estágio consta que o stream acabou, ela envia  nodos falsos (um nodo para cada thread) para o estágio 
 * seguinte.
 * 
 * Ao ler um nodo falso, as threads do segundo estágio passam o nodo para o terceiro estágio e encerram sua execução.
 * 
 * Ao ler um nodo falso, a thread do terceiro estágio encerra sua execução.
 * 
 * Como a thread do terceiro estágio espera receber apenas um nodo falso para encerrar a sua execução, os demais nodos
 * falsos são desalocados posteriormente, de maneira sequencial.
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
#include <pthread.h>
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
int quantidade_de_frames_na_fila;
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
pthread_mutex_t input_work_queue_lock;
pthread_mutex_t output_work_queue_lock;
pthread_cond_t frame_to_process;
pthread_cond_t frame_to_send;
pthread_t thread_stage_one;
pthread_t* threads_stage_two;
pthread_t thread_stage_three;

//protótipos de funções
int get_head_id_from_input_work_queue();
int get_head_id_from_output_work_queue();
void add_to_input_work_queue(work_node** node_aux);
void add_to_output_work_queue(work_node** node_aux);
work_node* remove_from_input_work_queue();
work_node* remove_from_output_work_queue();
void process_frame(work_node** node_aux);
void send_frame_to_display(work_node** node_aux);
void* stage_one(void* thread_argument);
void* stage_two(void* thread_argument);
void* stage_three(void* thread_argument);
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

	free(frame_aux);
	free(*node_aux);
}  

/**
 * Pós-condições: 
 * -Efetua leitura de todos os frames e os adiciona na fila encadeada de entrada.
 * -Cada vez que é adicionado um nodo de trabalho na fila de entrada, um sinal é
 * enviado para as threads do segundo estágio.
 * -Cada vez que um frame é lido, a variável nframes é incrementada.
 * -A função termina a sua execução quando todos os frames são terminados de serem
 * enviados.
 * -Quando é detectado o término da stream, o estágio deve criar e enviar nodos
 * "falsos" cujo único propósito é informar que a stream acabou, colocando em true
 * o valor da variável is_the_last_node. Devem ser gerados n nodos falsos, sendo n
 * o número de threads.
 */
void stage_one()
{
	while(1)	
	{
		Mat image;
		capture >> image;

		//atualiza o id que será utilizado pelo próximo frame
		nframes++;

		//se o frame capturado é vazio, envia nodos falsos que sinalizam o término da stream
		if (image.empty())
		{
			for(int i = 0; i < threads_number; i++)
			{
				//prepara o nodo de trabalho para ser enviado para a fila de entrada
				work_node* node_aux =  (work_node*) malloc(sizeof(work_node));	
				node_aux->frame = NULL;	
				node_aux->frame_number = nframes;	
				node_aux->next = NULL;			
				node_aux->is_the_last_node = true;

				//aplica o lock, adiciona o nodo na fila de entrada, manda o sinal e retira o lock
				pthread_mutex_lock(&input_work_queue_lock);
				add_to_input_work_queue(&node_aux);
				pthread_cond_signal(&frame_to_process);
				pthread_mutex_unlock(&input_work_queue_lock);
			}			

			break;
		}
		//envia um nodo de trabalho padrão para o processamento do segundo estágio			
		else
		{
			//prepara o nodo de trabalho para ser enviado para a fila de entrada
			work_node* node_aux =  (work_node*) malloc(sizeof(work_node));		
			node_aux->frame = new cv::Mat;
			(*(node_aux->frame)) = image.clone();			
			node_aux->frame_number = nframes;	
			node_aux->next = NULL;			
			node_aux->is_the_last_node = false;			

			//aplica o lock, adiciona o nodo na fila de entrada, manda o sinal e retira o lock
			pthread_mutex_lock(&input_work_queue_lock);
			add_to_input_work_queue(&node_aux);
			pthread_cond_signal(&frame_to_process);
			pthread_mutex_unlock(&input_work_queue_lock);
		}	
	}

	return;
} 

/**
 * Pós-condições: 
 * -Remove um nodo de trabalho da fila de entrada, o processa, o envia para a fila
 * de saída e envia um sinal para a thread do terceiro estágio. Esse procedimento
 * é executado até que se chegue na condição de encerramento da função.
 * -Caso a fila de entrada esteja vazia e ainda não se tenha chegado na condição
 * de encerramento da função, a thread deve esperar pelo sinal frame_to_process.
 * -A condição de encerramento do segundo estágio é quando todos os frames do
 * vídeo já foram enviados para a fila de entrada e já foram processados no segundo
 * estágio, isso é verificado através do valor da variável is_the_last_node no nodo
 * de trabalho sendo true. Note que apenas nodos falsos apresentam esse valor nessa
 * variável e os mesmos não devem ser processados pela função process_frame.
 */
void stage_two()
{
	while(1)
	{
		//aplica lock na fila de entrada para acessar elementos
		pthread_mutex_lock(&input_work_queue_lock);

		//espera receber sinal se a fila de entrada estiver vazia
		while(size_input_work_queue == 0)
		{
			pthread_cond_wait(&frame_to_process, &input_work_queue_lock);			
		}

		//remove o nodo da fila de entrada e retira o lock
		work_node* node_aux = remove_from_input_work_queue();
		pthread_mutex_unlock(&input_work_queue_lock);

		//se este é um dos nodos que sinaliza o término da stream, envia o nodo para o próximo estágio e encerra a execução
		if((node_aux->is_the_last_node) == true)
		{			
			//aplica o lock, adiciona o nodo na fila de saída, manda o sinal e retira o lock
			pthread_mutex_lock(&output_work_queue_lock);
			add_to_output_work_queue(&node_aux);
			pthread_cond_signal(&frame_to_send);
			pthread_mutex_unlock(&output_work_queue_lock);

			break;			
		}
		//envia um nodo de trabalho padrão para o processamento do terceiro estágio
		else
		{
			//processa o frame, aplica o lock, adiciona o nodo na fila de saída, manda o sinal e retira o lock
			process_frame(&node_aux);
			pthread_mutex_lock(&output_work_queue_lock);
			add_to_output_work_queue(&node_aux);
			pthread_cond_signal(&frame_to_send);
			pthread_mutex_unlock(&output_work_queue_lock);
		}	


	}	

	return;
} 

/**
 * Pós-condições: 
 * -Verifica se o primeiro nodo da fila de saída é o nodo correto para se enviar ao
 * display, se sim, remove o nodo de trabalho da fila e o envia para o display, caso
 * contrário, a thread espera por um sinal. Esse procedimento é executado até que se 
 * chegue na condição de encerramento da função.
 * -A condição de encerramento do terceiro estágio é quando todos os frames do vídeo 
 * já foram enviados para a fila de entrada, já foram processados no segundo estágio
 * e já foram enviados para o display, isso é verificado através do valor da variável 
 * (is_the_last_node) do nodo de trabalho retirado da fila. Se a variável is_the_last_node
 * possui o valor true, isso significa que este é o nodo falso que sinaliza o término da 
 * stream e então a thread pode ser encerrada. Note que o terceiro estágio faz leitura de
 * apenas um nodo falso e encerra a sua execução. Os nodos falsos restantes são desalocados
 * mais adiante no código, quando já não se precisa controles de lock por exemplo, o que
 * economiza operações.
 */
void stage_three()
{
	while(1)
	{
		//atualiza o id do frame requerido 
		current_frame++;

		//aplica lock na fila de saída para acessar elementos
		pthread_mutex_lock(&output_work_queue_lock);

		//espera receber sinal se a fila de saída estiver vazia
		while(size_output_work_queue == 0)
		{
			pthread_cond_wait(&frame_to_send, &output_work_queue_lock);
		}
		//espera receber sinal se o primeiro elemento da fila é diferente do frame necessário
		while(get_head_id_from_output_work_queue() != current_frame)
		{
			pthread_cond_wait(&frame_to_send, &output_work_queue_lock);
		}

		//remove o nodo da fila e retira o lock
		work_node* node_aux = remove_from_output_work_queue();		
		pthread_mutex_unlock(&output_work_queue_lock);

		//se é o nodo que sinaliza o final da stream, cancela a execução da thread
		if((node_aux->is_the_last_node) == true)
		{
			break;			
		}

		//manda o frame para o display
		send_frame_to_display(&node_aux);	
	}

	return;
} 

/**
 * Pós-condições: 
 * Executa o algoritmo usando processamento paralelo.
 */
void parallel_processing() 
{
	is_there_any_work = true;
	finished = false;
	
	//quantidade de threads que serão criadas pelo openmp
	omp_set_num_threads(threads_number);

	omp_lock_t lock_capturar_frame; 
	omp_init_lock(&lock_capturar_frame);

	omp_lock_t lock_da_fila_de_saida; 
	omp_init_lock(&lock_da_fila_de_saida);

	//execução com openmp
	#pragma omp parallel shared(is_there_any_work, finished)
	while(finished == false)
	{
		work_node* node_aux;	
		int id_of_my_frame;	
		Mat image;

		#pragma omp single nowait
		{
			//####//ATIVA LOCK DA FILA DE SAIDA//####//
			omp_set_lock(&lock_da_fila_de_saida);

			if(size_output_work_queue > 0)
			{
				int id_of_the_head_of_the_queue = get_head_id_from_output_work_queue();

				if(id_of_the_head_of_the_queue == current_frame)
				{
					node_aux = remove_from_output_work_queue();

					//####//DESATIVA LOCK DA FILA DE SAIDA//####//
					omp_unset_lock(&lock_da_fila_de_saida);

					if((node_aux->is_the_last_node) == true)
					{
						finished = true;	
					}
					else
					{						
						//manda o frame para o display
						send_frame_to_display(&node_aux);

						//printf("FRAME ENVIADO PARA O DISPLAY: %d\n", current_frame);

						current_frame++;
					}
				}	
				else
				{
					//####//DESATIVA LOCK DA FILA DE SAIDA//####//
					omp_unset_lock(&lock_da_fila_de_saida);			
				}	
			}
			else
			{
				//####//DESATIVA LOCK DA FILA DE SAIDA//####//
				omp_unset_lock(&lock_da_fila_de_saida);
			}
		}	

		if(is_there_any_work == true)
		{	

			//####//ATIVA LOCK DE CAPTURA de FRAMES//####//
			omp_set_lock(&lock_capturar_frame);

			capture >> image;

			if (image.empty()) 
			{
				id_of_my_frame = nframes;
				nframes++;

				//####//DESATIVA LOCK DE CAPTURA DE FRAMES//####//
				omp_unset_lock(&lock_capturar_frame);

				work_node* node_aux =  (work_node*) malloc(sizeof(work_node));	
				node_aux->frame = NULL;	
				node_aux->frame_number = id_of_my_frame;	
				node_aux->next = NULL;			
				node_aux->is_the_last_node = true;

				//####//ATIVA LOCK DA FILA DE SAIDA//####//
				omp_set_lock(&lock_da_fila_de_saida);

				add_to_output_work_queue(&node_aux);

				//####//DESATIVA LOCK DA FILA DE SAIDA//####//
				omp_unset_lock(&lock_da_fila_de_saida);

				is_there_any_work = false;
			}
			else
			{
				id_of_my_frame = nframes;
				nframes++;

				//####//DESATIVA LOCK DE CAPTURA DE FRAMES//####//
				omp_unset_lock(&lock_capturar_frame);

				node_aux =  (work_node*) malloc(sizeof(work_node));		
				node_aux->frame = new cv::Mat;
				(*(node_aux->frame)) = image.clone();			
				node_aux->frame_number = id_of_my_frame;	
				node_aux->next = NULL;
				node_aux->is_the_last_node = false;

				//####//PROCESSA FRAME//####//
				process_frame(&node_aux);

				//####//ATIVA LOCK DA FILA DE SAIDA//####//
				omp_set_lock(&lock_da_fila_de_saida);

				add_to_output_work_queue(&node_aux);

				//####//DESATIVA LOCK DA FILA DE SAIDA//####//
				omp_unset_lock(&lock_da_fila_de_saida);
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
	current_frame = 0;
	nframes = 0;

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
	oVideoWriter.open("result_openmp.avi", CV_FOURCC('P','I','M','1'), 20, frameSize, true); 	

	auto tstart = std::chrono::high_resolution_clock::now();

	parallel_processing();	

	auto tend = std::chrono::high_resolution_clock::now();

	double TT; //TIME
	TT = std::chrono::duration<double>(tend-tstart).count();
	double TR = nframes/TT; //FRAMES

	cout << "EXECUTION TIME IN SECONDS: " << TT << endl;
	cout << "FRAMES PER SECOND: " << TR << endl;

	return 0;
}
