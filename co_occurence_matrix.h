/*
 * 8bit画像の同時生起行列を求める関数
 * r=1, theta=0,45,90,135
 * 同時生起行列は対象行列となる．
 * 高木，下田，画像解析ハンドブック，東京大学出版，pp518-521，1991．
 */

#define COM_DTYPE_32F (float)1.0
#define COM_DTYPE_64F (double)1.0
#define CAL_EIGEN_VALUE_OFF 100
#define CAL_EIGEN_VALUE_ON  101


int
CoOccurrenceMatrix( cv::Mat &img ,// 8bit 1channel image 
				    cv::Mat comat[]);//出力(32S/32F/64F) [0]:0, [1]:45, [2]:90, [3]:135 


/*
 *  同時生起行列を同時生起確率に変換する．
 *  入力(32S/32F/64F) と出力の型は，一致する．
 *  ただし，入力が32Sのときは，出力は32F．
 */
int
TransformToProbability( cv::Mat &comat, // 入力(32S/32F/64F) 
						cv::Mat &dist);// 出力(32F/64F)  


/*
 * 同時生起確率から計算される14の特徴量
 *   <float> or <double>
 *  計算しない場合 14番目 features[13]=-1 となる．
 */
int
FeaturesOfCoOccurrenceProbability(cv::Mat &comatP,   // 同時生起確率
								  cv::Mat &features, // 出力：14の特徴量
								  int calc_eigen);   // 特徴量14番目の固有値計算するかどうか



