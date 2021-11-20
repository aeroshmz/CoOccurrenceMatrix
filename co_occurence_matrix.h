/*
 * 8bit�摜�̓������N�s������߂�֐�
 * r=1, theta=0,45,90,135
 * �������N�s��͑Ώۍs��ƂȂ�D
 * ���؁C���c�C�摜��̓n���h�u�b�N�C������w�o�ŁCpp518-521�C1991�D
 */

#define COM_DTYPE_32F (float)1.0
#define COM_DTYPE_64F (double)1.0
#define CAL_EIGEN_VALUE_OFF 100
#define CAL_EIGEN_VALUE_ON  101


int
CoOccurrenceMatrix( cv::Mat &img ,// 8bit 1channel image 
				    cv::Mat comat[]);//�o��(32S/32F/64F) [0]:0, [1]:45, [2]:90, [3]:135 


/*
 *  �������N�s��𓯎����N�m���ɕϊ�����D
 *  ����(32S/32F/64F) �Əo�͂̌^�́C��v����D
 *  �������C���͂�32S�̂Ƃ��́C�o�͂�32F�D
 */
int
TransformToProbability( cv::Mat &comat, // ����(32S/32F/64F) 
						cv::Mat &dist);// �o��(32F/64F)  


/*
 * �������N�m������v�Z�����14�̓�����
 *   <float> or <double>
 *  �v�Z���Ȃ��ꍇ 14�Ԗ� features[13]=-1 �ƂȂ�D
 */
int
FeaturesOfCoOccurrenceProbability(cv::Mat &comatP,   // �������N�m��
								  cv::Mat &features, // �o�́F14�̓�����
								  int calc_eigen);   // ������14�Ԗڂ̌ŗL�l�v�Z���邩�ǂ���



