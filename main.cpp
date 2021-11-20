#define _CRT_SECURE_NO_WARNINGS 


#include <Windows.h>
#include <stdio.h>
#include <iostream>
#include <vector>

//OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "../cv2util/cv2util.h"
#include "co_occurence_matrix.h"



int
main(int argc, char *argv[])
{

	//TESTデータの読み込み
	printf("--------testdata-------\n");
	cv::Mat testdata=LoadMatrix("testdata.xml" ,//ファイル名
						"data");		//読み込む行列のタグ
	
	PrintMat(testdata, "%d\t");
	printf("-----------------------\n");

	printf("\n");
	//printf("******* testdata ******\n");
	//cv::Mat testdata = ( cv::Mat_<double>(3,2) << 1.1, 2.3, 3.6, 4.3, 5.4, 6.3);
	//PrintMatType(testdata);
	//std::cout<< "testdata.rows=" << testdata.rows << std::endl;
	//std::cout<< "testdata.cols=" << testdata.cols << std::endl;
	//std::cout<< "testdata.step=" << testdata.step << std::endl;
	//std::cout<< "testdata.dims=" << testdata.dims << std::endl;
	//std::cout<< "testdata.channels()=" << testdata.channels() << std::endl;


	//[1]濃度同時生起行列の作成
	cv::Mat comat[4]; //濃度同時生起行列
	cv::Mat comatP[4];//濃度同時生起確率

	//[2]濃度同時生起行列の初期化
	for (int i = 0; i < 4; i++) {
		comat[i] = cv::Mat::zeros(4, 4, CV_32S);
		comatP[i] = cv::Mat::zeros(4, 4, CV_32F);
	}

	//[3]濃度同時生起行列の計算
	int64 time0 = cv::getTickCount();
	CoOccurrenceMatrix( testdata , 
					    comat);//出力 [0]:0, [1]:45, [2]:90, [3]:135

	int64 time1 = cv::getTickCount();
    std::cout << "co-occurence matrix calculation time   "<<(time1-time0)/cv::getTickFrequency() << "s" << std::endl;

	//濃度同時生起行列の表示
	for(int k=0;k<4;k++){
		printf("***** %d degree *****\n",k*45);
		PrintMat(comat[k], "%d\t");	
	}

	//[4] 同時生起行列を同時生起確率に変換する．
	/*  入力(32S/32F/64F) と出力の型は，一致する．
	 *  ただし，入力が32Sのときは，出力は32F．
	 */
	for(int k=0;k<4;k++)
		TransformToProbability(comat[k],// 入力(32S/32F/64F) 
						       comatP[k]);// 出力(32F/64F)  

	//濃度同時生起確率の表示
	for(int k=0;k<4;k++){
		printf("***** %d degree(R) *****\n",k*45);
		PrintMat(comatP[k], "%3.3f\t");
		printf("----------\n");
	}

	/*
	 * 同時生起確率から計算される14の特徴量
	 *
	 */
	int64 time2 = cv::getTickCount();
	cv::Mat features[4];
	for ( int k=0; k<4 ; k++ ){
		features[k]=cv::Mat::zeros(14, 1, CV_32F);
		FeaturesOfCoOccurrenceProbability( comatP[k], features[k] , CAL_EIGEN_VALUE_ON );
	}
	int64 time3 = cv::getTickCount();
    std::cout << "14 features calculation time   "<<(time3-time2)/cv::getTickFrequency()<< "s" << std::endl;
	std::cout <<std::endl;

	for(int n=0;n<4;n++){
	float *pf = features[n].ptr<float>(0);
		for (int k = 0; k < 14; k++) {
			printf("%d degree features[%d] = %f \n",n*45 , k, pf[k] );
		}
		printf("\n");
	}


	//濃度同時生起行列の保存
    cv::FileStorage fs("comat.xml", cv::FileStorage::WRITE);
    if (!fs.isOpened()){
        std::cout << "File can not be opened." << std::endl;
        return -1;
    }
	for (int k = 0; k < 4; k++) {
		char tag[32];
		//sprintf(tag, "%d degree",k*45);
		fs << "angle" << k*45;
		sprintf(tag, "comat%d",k*45);
		fs << tag << comat[k];
		sprintf(tag, "comat%dP",k*45);
		fs << tag << comatP[k];
		sprintf(tag, "features%d", k*45);
		fs << tag << features[k];
	}

	//書き出し
    fs.release();


	return 1;
}