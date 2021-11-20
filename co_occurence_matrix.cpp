/*
 * 8bit�摜�̓������N�s������߂�֐�
 * r=1, theta=0,45,90,135
 * �������N�s��͑Ώۍs��ƂȂ�D
 * ���؁C���c�C�摜��̓n���h�u�b�N�C������w�o�ŁCpp518-521�C1991�D
 */

//debug����Ƃ��C�R�����g�A�E�g
//#define _CODEBUG_   

#include <opencv2/opencv.hpp>
#include <iostream>

//�ǂ��炩����
//#define USE_CV_EIGEN  /* OpenCV�Ɏ�������Ă���ŗL�l�v�Z���g���ꍇ */
#define USE_EIGEN3    /* EIGEN3�𗘗p���ČŗL�l�v�Z����ꍇ */


#ifdef USE_EIGEN3
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#endif

//�ŗL�l�̍ŏ��l
#define EIGENEPS 1.0e-6

#include "..\cv2util\cv2util.h"
#include "co_occurence_matrix.h"


// �f�o�b�O�p
#define _CODEBUG_ 


template <typename Type>
int
CoOccurrenceMatrix( cv::Mat &img ,// 8bit 1channel image 
				    cv::Mat comat[],//�o��(32S/32F/64F) [0]:0, [1]:45, [2]:90, [3]:135 
					Type n)       // n�͌^���L���X�g�œn���Ή��ł��悢�D

{
	//�f�[�^�|�C���^
	uchar *val[9];
	//  0       1       2
	//  3(j)  (u,v)(i)  5(j)
	//  6       7       8
	for (int v = 0; v < img.rows; v++) {
		for (int u = 0; u < img.cols; u++) {
			//����8��f�̃|�C���^���쐬
			int k = 0;
			for (int j = v - 1; j<=v + 1; j++) {
				for (int i = u - 1; i <= u + 1; i++, k++) {
					if( isInMat(img, i, j) ==1 ) val[k] = img.ptr(j)+i;
					else val[k] = NULL;
				}
			}
			//0��
			if (val[5] != NULL) comat[0].ptr<Type>(*val[4])[ *val[5] ] ++;
			if (val[3] != NULL) comat[0].ptr<Type>(*val[4])[ *val[3] ] ++;
			//45��
			if (val[2] != NULL) comat[1].ptr<Type>(*val[4])[ *val[2] ] ++;
			if (val[6] != NULL) comat[1].ptr<Type>(*val[4])[ *val[6] ] ++;
			//90��
			if (val[1] != NULL) comat[2].ptr<Type>(*val[4])[ *val[1] ] ++;
			if (val[7] != NULL) comat[2].ptr<Type>(*val[4])[ *val[7] ] ++;
			//135��
			if (val[0] != NULL) comat[3].ptr<Type>(*val[4])[ *val[0] ] ++;
			if (val[8] != NULL) comat[3].ptr<Type>(*val[4])[ *val[8] ] ++;

		}
	}


#ifdef _CODEBUG_
	for(int k=0;k<4;k++){
		printf("***** %d degree *****\n",k*45);
		for( int v=0; v<comat[k].rows; v++ ){
			for( int u=0; u<comat[k].cols; u++ ){
				printf("%d\t", (int)comat[k].ptr<Type>(v)[u] );
			}
			printf("\n");
		}
	}
#endif

	return 1;
}

int
CoOccurrenceMatrix( cv::Mat &img ,// 8bit 1channel image 
				    cv::Mat comat[])//�o��(32S/32F/64F) [0]:0, [1]:45, [2]:90, [3]:135 
{
	if (img.type()!=CV_8U) {
		printf("CoOccurenceMatrix()\n");
		printf("  Input image is not CV_8U.\n");
		return -1;
	}
	if ( !(comat[0].type()==CV_32S ||
		 comat[0].type()==CV_32F ||
		 comat[0].type()==CV_64F )){
		printf("CoOccurenceMatrix()\n");
		printf("  co occurence matrix must be CV_32S or CV_32F or CV_64F.\n");
		return -1;
	}

	if (comat[0].type() == CV_32S) {
		CoOccurrenceMatrix( img , comat , (int) 1);
	}else if (comat[0].type() == CV_32F) {//���N�s��̃^�C�v
		CoOccurrenceMatrix( img , comat, (float) 2.0);
	} else if (comat[0].type() == CV_64F) {
		CoOccurrenceMatrix( img , comat, (double)3.0);
	}


	return 1;
}
/*
 *  �������N�s��𓯎����N�m���ɕϊ�����D
 *  ����(32S/32F/64F) �Əo�͂̌^�́C��v����D
 *  �������C���͂�32S�̂Ƃ��́C�o�͂�32F�D
 */
int
TransformToProbability(cv::Mat &comat,// ����(32S/32F/64F) 
					   cv::Mat &dist)// �o��(32F/64F)  
{
	
	//���a�̌v�Z
	cv::Scalar val=cv::sum(comat);
	double sum=val.val[0];
	
	//comat --> dist  dist.type�ŃR�s�[
	comat.convertTo(dist,dist.type(),1.0);

	//���a�Ŋ���D
	dist/=sum;

	return 1;
}

/*
 * �������N�m������v�Z�����14�̓�����
 *   <float> or <double>
 *  �v�Z���Ȃ��ꍇ 14�Ԗ� features[13]=-1 �ƂȂ�D
 */
template< typename type >
int
FeaturesOfCoOccurrenceProbability(cv::Mat &comatP ,  // �������N�m��
								  cv::Mat &features, // �o�́F14�̓�����
								  const int cv_type, // CV_32F or CV_64F
								  type flag,         // COM_DTYPE_32F or COM_DTYPE_64F
								  int calc_eigen)    // ������14�Ԗڂ̌ŗL�l�v�Z���邩�ǂ���
{
#ifdef _CODEBUG_
	printf("In FeaturesOfCoOccurrenceProbability()\n");
	PrintMat(comatP, "%3.3e\t");
	printf("=============\n");
#endif

	cv::Mat Px=cv::Mat::zeros(comatP.rows,1,cv_type);//Px(i)
	cv::Mat Py=cv::Mat::zeros(comatP.cols,1,cv_type);//Py(j) reduce�ŋ��߂���[ cols,1 ]�ɕό`
	cv::Mat Pxpy=cv::Mat::zeros(comatP.rows+comatP.cols,1,cv_type);//Px+y(k)
	cv::Mat Px_y=cv::Mat::zeros(comatP.rows>comatP.cols ? comatP.rows : comatP.cols, 1, cv_type);//Px-y(k)
	cv::Mat tmpMat;
	
	//************* 14�̓����ʂ̌v�Z���� ****************
	type val=0;

	//***********  Px�̌v�Z rows���ւ̎��Ӊ� ***********
	//ver.�n���Ɏ���
	//for (int i = 0; i < comatP.rows;i++){
	//	val=0;
	//	for (int j = 0; j < comatP.cols; j++){
	//		val+=(type)comatP.ptr<type>(i)[j];
	//		//printf("j=%d %f\n", j, comatP.ptr<typet>(j)[i] );
	//	}
	//	Px.ptr<type>(i)[0]=val;
	//}
	/*
	dim = 0 �̏ꍇ�A�s�����ɏ������s���A��݂̂ƂȂ�
	dim = 1 �̏ꍇ�A������ɏ������s���A�s�݂̂ƂȂ�
	rtype
	cv::REDUCE_SUM ���v
	cv::REDUCE_AVG ����
	cv::REDUCE_MAX �ő�l
	cv::REDUCE_MIN �ŏ��l
	*/
	cv::reduce(comatP, Px, 1, cv::REDUCE_SUM, cv_type);

#ifdef _CODEBUG_
	printf("*********** Px ***********\n");
	for (int k = 0; k<Px.rows; k++) 
		printf("Px[%d]=%f\n", k, Px.ptr<type>(0)[k] );
#endif

	//***********  Py�̌v�Z cols���ւ̎��Ӊ� ***********
	//ver.�n���Ɏ���
	//for (int j = 0; j < comatP.cols; j++){
	//	val=0;
	//	for (int i = 0; i < comatP.rows;i++){
	//		val+=comatP.ptr<type>(i)[j];
	//		//printf("i=%d %f\n", i, comatP.ptr<type>(j)[i] );
	//	}
	//	Py.ptr<type>(j)[0]=val;
	//	//printf("j=%d  val=%3.3f \n", j , val );
	//}
	cv::reduce(comatP, Py, 0, cv::REDUCE_SUM, cv_type);
	Py = Py.t();

#ifdef _CODEBUG_
	printf("*********** Py ***********\n");
	for (int k = 0; k<Py.rows; k++) 
		printf("Py[%d]=%f\n", k, Py.ptr<type>(0)[k] );
#endif
	//***********  Px+y�̌v�Z  ***********
	for (int k = 0;k < Pxpy.rows;k++) {
		val=0;
		for (int i = 0; i < comatP.rows; i++){
			for (int j = 0; j < comatP.cols;j++){
				if ( i+j==k ) val+=comatP.ptr<type>(i)[j];
			}
		}
		Pxpy.ptr<type>(k)[0]=val;
		//printf("j=%d  val=%3.3f \n", j , val );
	}
#ifdef _CODEBUG_
	printf("*********** Px+y ***********\n");
	for (int k = 0; k<Pxpy.rows; k++) 
		printf("Px+y[%d]=%f\n", k, Pxpy.ptr<type>(0)[k] );
#endif

	//***********  Px-y�̌v�Z ***********
	for (int k = 0;k < Px_y.rows;k++) {
		val=0;
		for (int i = 0; i< comatP.rows; i++){
			for (int j = 0; j < comatP.cols;j++){
				if ( abs(i-j)==k ) 	val+=comatP.ptr<type>(i)[j];
			}
		}
		Px_y.ptr<type>(k)[0]=val;
	}
#ifdef _CODEBUG_
	printf("*********** Px_y ***********\n");
	for (int k = 0; k<Px_y.rows; k++) 
		printf("Px-y[%d]=%f\n", k, Px_y.ptr<type>(0)[k] );
#endif

	//[1] angular second moment (2.2.8)
	tmpMat=comatP.mul(comatP);
	cv::Scalar scval=cv::sum(tmpMat);
	double sum=scval.val[0];

	*features.ptr<type>(0)=sum;//val


	//[2] contrast (2.2.9)
	val=0;
	for ( int k=0; k<Px_y.rows; k++ ) val+=k*k*Px_y.ptr<type>(k)[0];
	*features.ptr<type>(1)=val;


	//[3] correlation (2.2.10)
	type mu_x=0, mu_y=0, sigma_x2=0, sigma_y2=0;
	for (int i = 0;i < Px.rows; i++) mu_x += i*Px.ptr<type>(i)[0];
	for (int i = 0;i < Px.rows; i++) sigma_x2 += (i-mu_x)*(i-mu_x)*Px.ptr<type>(i)[0];
	for (int j = 0;j < Py.rows; j++) mu_y += j*Py.ptr<type>(j)[0];
	for (int j = 0;j < Py.rows; j++) sigma_y2 += (j-mu_y)*(j-mu_y)*Py.ptr<type>(j)[0];
#ifdef _CODEBUG_
	printf("mu_x=%f\n",mu_x);
	printf("mu_y=%f\n",mu_x);
	printf("sigma_x2=%f\n",sigma_x2);
	printf("sigma_y2=%f\n",sigma_y2);
#endif
	val=0;
	for (int i = 0;i < comatP.rows;i++) {
		for ( int j=0; j<comatP.cols; j++){
			val+=i*j*comatP.ptr<type>(i)[j];
		}
	}
	val = ( val - mu_x*mu_y ) / sqrt(sigma_x2*sigma_y2);
	*features.ptr<type>(2)=val;

	//[4] sum of suqare : variance (2.2.11)
	val=0;
	for (int i = 0;i < comatP.rows;i++) {
		for ( int j=0; j<comatP.cols; j++){
			val+=(j-mu_x)*(j-mu_x)*comatP.ptr<type>(i)[j];
		}
	}
	*features.ptr<type>(3)=val;

	//[5] inverse difference moment (2.2.12)
	val=0;
	for (int i = 0;i < comatP.rows;i++) {
		for ( int j=0; j<comatP.cols; j++){
			val+=1.0/(1.0+(i-j)*(i-j))*comatP.ptr<type>(i)[j];
		}
	}
	*features.ptr<type>(4)=val;

	//[6] sum average (2.2.13)
	val=0;
	for (int k = 0;k < Pxpy.rows;k++) {
		val+=k*Pxpy.ptr<type>(k)[0];
	}
	*features.ptr<type>(5)=val;

	//[7] sum variance (2.2.14)
	val=0;
	for (int k = 0;k < Pxpy.rows;k++) {
		val+=(k- (*features.ptr<type>(5)) )*(k-(*features.ptr<type>(5)))*Pxpy.ptr<type>(k)[0];
	}
	*features.ptr<type>(6)=val;

	//[8] sum entropy (2.2.15)  log�̒��2
	val=0;
	for (int k = 0;k < Pxpy.rows;k++) {
		val+=-Pxpy.ptr<type>(k)[0] * ( Pxpy.ptr<type>(k)[0]==0 ? 0:  log( Pxpy.ptr<type>(k)[0] )/log(2)    );
	}
	*features.ptr<type>(7)=val;

	//[9] entropy (2.2.16)  log�̒��2
	val=0;
	for (int i = 0;i< comatP.rows;i++) {
		for ( int j=0; j<comatP.cols; j++){
			val += -comatP.ptr<type>(i)[j] * ( comatP.ptr<type>(i)[j]==0 ? 0 :  log(comatP.ptr<type>(i)[j])/log(2)  );
		}
	}
	*features.ptr<type>(8)=val;
	//comatP.copyTo(tmpMat);
	//cv::Mat tmpMatLog;
	//tmpMat.setTo(1,comatP==0);//because log(1)=0
	//cv::log(tmpMat,tmpMatLog);
	//tmpMat = comatP.mul(tmpMatLog/log(2));
	//scval = cv::sum(tmpMat);
	//sum = -scval.val[0];
	//*features.ptr<type>(8)=sum;



	//[10] difference variance (2.2.17)
	val=0;
	for (int k = 0;k < Px_y.rows;k++) val += k*Px_y.ptr<type>(k)[0];
	*features.ptr<type>(9)=val;
	val=0;
	for (int k = 0;k < Px_y.rows;k++) {
		val += ( k - (*features.ptr<type>(9)) )*( k - (*features.ptr<type>(9)) )*Px_y.ptr<type>(k)[0];
	}
	*features.ptr<type>(9)=val;

	//[11] difference entropy (2.2.18)
	val=0;
	for (int k = 0;k < Px_y.rows;k++) val += -Px_y.ptr<type>(k)[0]* ( Px_y.ptr<type>(k)[0]==0 ? 0: log(Px_y.ptr<type>(k)[0])/log(2)  );
	*features.ptr<type>(10)=val;

	//[12] information measure of correlation (2.2.19)
	type HXY=0, HX=0, HY=0, HXY1=0, HXY2=0;
	for (int i = 0; i < comatP.rows;i++) {
		for (int j = 0; j < comatP.cols;j++) {
			//HXY  += -comatP.ptr<type>(i)[j] * ( comatP.ptr<type>(i)[j]==0 ? 0 : log(comatP.ptr<type>(i)[j])/log(2) );
			HXY1 += -comatP.ptr<type>(i)[j] * (  Px.ptr<type>(i)[0]*Py.ptr<type>(j)[0] ==0 ? 0 : log( Px.ptr<type>(i)[0]*Py.ptr<type>(j)[0] )/log(2) );
			HXY2 += -Px.ptr<type>(i)[0] * Py.ptr<type>(j)[0] * ( Px.ptr<type>(i)[0] * Py.ptr<type>(j)[0] == 0 ? 0 : log(Px.ptr<type>(i)[0]*Py.ptr<type>(j)[0])/log(2)  );
		}
		HX += -Px.ptr<type>(i)[0] * ( Px.ptr<type>(i)[0] ==0 ? 0 : log(Px.ptr<type>(i)[0])/log(2) );
		HY += -Py.ptr<type>(i)[0] * ( Py.ptr<type>(i)[0] ==0 ? 0 : log(Py.ptr<type>(i)[0])/log(2) );
	}
	HXY=*features.ptr<type>(8);
#ifdef _CODEBUG_
	printf("HXY=%f\n",HXY);
	printf("HX=%f \n",HX);
	printf("HY=%f \n",HY);
	printf("HXY1=%f\n",HXY1);
	printf("HXY2=%f\n",HXY2);
#endif

	*features.ptr<type>(11) = ( HXY - HXY1 ) / ( HX > HY ? HX : HY );

	//[13] information measure of correlation (2.2.20)
	val = 1-exp( -2.0*(HXY2-HXY) );
	*features.ptr<type>(12) = ( val >= 0 ? sqrt(val) : -1 );
	
	//�����܂ł�0.13s������(�S�̂ł�0.82s)

	//[14] maximal correlation coefficient (2.2.21)
	//     (Q��2�Ԗڂɑ傫���ŗL�l)^0.5
	// �������N�s��͐��̑Ώۍs��Ȃ̂ŁC���Ȃ��Ƃ�������l
	cv::Mat Q( comatP.size() , cv_type );
	// Q�쐬
	for (int i = 0; i < comatP.rows;i++) {
		for (int j = 0; j < comatP.cols;j++) {
			val=0;
			for( int k=0; k< comatP.cols; k++ ){
				val += comatP.ptr<type>(i)[k] * comatP.ptr<type>(k)[j] ;
			}
			val = ( Px.ptr<type>(i)[0] * Py.ptr<type>(j)[0] == 0? 0 : val/( Px.ptr<type>(i)[0] * Py.ptr<type>(j)[0]) );
			Q.ptr<type>(i)[j]= val;
		}
	}

#ifdef _CODEBUG_
	std::cout << "***** Q *****\n";
	PrintMat(Q, "%3.3f\t");
#endif


if( calc_eigen == CAL_EIGEN_VALUE_ON ){
	//�ŗL�l�v�Z�����Ԃ�������
	//�S�̂�3/4% ���x(cv::eigen) �����ȑΏۍs��̕K�v����D
	//------------------------------------------------------
	#ifdef USE_CV_EIGEN 
		cv::Mat QeigenVectors( comatP.size() , cv_type );
		cv::Mat QeigenValues(  comatP.rows, 1, cv_type );

		//int64 timeS = cv::getTickCount();
	    //cv::eigen(Qd,QeigenValues,QeigenVectors);
	    cv::eigen(Q,QeigenValues,QeigenVectors);

		//int64 timeE = cv::getTickCount();
	    //std::cout << "cv::eigen method calculation time   "<<(timeE-timeS)/cv::getTickFrequency()<< "s" << std::endl;

		if( fabs(QeigenValues.ptr<type>(1)[0])<EIGENEPS ) *features.ptr<type>(13)=0; //Q��2�Ԗ�
		else *features.ptr<type>(13) = sqrt(QeigenValues.ptr<type>(1)[0]);

	#endif

	//------------------------------------------------------
	#ifdef USE_EIGEN3
		Eigen::MatrixXd emat;

		#ifdef _CODEBUG_
		int64 timeS = cv::getTickCount();
		#endif
		
		cv::cv2eigen(Q, emat);
		
		Eigen::EigenSolver<Eigen::MatrixXd> es(emat);
		if (es.info() != Eigen::Success){
			std::cout<<"Eigen value calculation error\n";
			abort();
		}
		
		//std::cout << "�ŗL�l�F\n" << es.eigenvalues() << std::endl;
		//std::cout << "�ŗL�x�N�g���F\n"	<< es.eigenvectors() << std::endl;

		//sort
		std::vector<double> v;
		for(int i=0; i<es.eigenvalues().rows(); i++){
			v.push_back( es.eigenvalues()(i).real()  );
		}
		
		//std::sort( v.begin(), v.end() );//, sizeof(double), compare_head );
		std::sort( v.begin(), v.end(), 
			[](auto const& lhs, auto const& rhs) {
    			return lhs > rhs; // ���̕����傫��...�Ƃ����C���[�W�B
  			}
  		);
	#ifdef _CODEBUG_
		int64 timeE = cv::getTickCount();
		for(int i=0; i<v.size(); i++) printf("eigen value [%d] = %f\n", i, v[i] );
		for( int i=0; i<v.size(); i++ ){
			std::cout<<" eigen vector [ " << i << " ]= " << v[i] << "  sqrt:"<<sqrt(v[i])<<std::endl;
		}
		
	    std::cout << "Eigen3 EigenSolver calculation time   "<<(timeE-timeS)/cv::getTickFrequency()<< "s" << std::endl;
	#endif

		if( fabs(v[1])<EIGENEPS ) *features.ptr<type>(13)=0; //Q��2�Ԗ�
		else *features.ptr<type>(13) = sqrt(v[1]); 
		

	#endif
	//------------------------------------------------------


	}else{
		*features.ptr<type>(13) = -1;
	}
	
	return 1;
}


/*
 * �������N�m������v�Z�����14�̓�����
 *   <float> or <double>
 */
int
FeaturesOfCoOccurrenceProbability(cv::Mat &comatP,   // �������N�m��
								  cv::Mat &features, // �o�́F14�̓�����
								  int calc_eigen)    // ������14�Ԗڂ̌ŗL�l�v�Z���邩�ǂ���
{

	if (features.rows != 14) {
		printf(" cv::Mat &features size must be 14x1.\n");
		return 0;
	}


	if (comatP.type()==CV_32F && features.type() == CV_32F) { 
		FeaturesOfCoOccurrenceProbability(comatP,   // �������N�m��
									      features, // �o�́F14�̓�����
									      CV_32F,
									      COM_DTYPE_32F,
									      calc_eigen);    // ������14�Ԗڂ̌ŗL�l�v�Z���邩�ǂ���
									      
	} else if ( comatP.type()==CV_64F && features.type() == CV_64F ){
		FeaturesOfCoOccurrenceProbability(comatP,   // �������N�m��
									      features, // �o�́F14�̓�����
									      CV_64F,
									      COM_DTYPE_64F,
									      calc_eigen);    // ������14�Ԗڂ̌ŗL�l�v�Z���邩�ǂ���

	} else {
		printf("You must use same type matrix for FeaturesOfCoOccurrenceProbability().\n");
		printf(" CV_32F or CV_64F \n");
		return 0;
	}


	return 1;
}
