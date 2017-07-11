#include "necstream.hpp"
#define WIDTH 50
#define CH 3
#define N 256

typedef ap_uint<8> data;
typedef hls::stream<data> STREAM;
typedef hls::Mat<WIDTH, WIDTH, HLS_8UC1> IMAGE_C1;
typedef hls::Mat<WIDTH, WIDTH, HLS_8UC3> IMAGE_C3;

void MyEqualizer(data src[], data dst[], data hist[], data cdfMin);
void image_filter(UIntStream& src_axi, UIntStream& dst_axi) {

#pragma HLS INTERFACE axis port=dst_axi bundle = OUTPUT
#pragma HLS INTERFACE axis port=src_axi bundle = INPUT
#pragma HLS INTERFACE ap_ctrl_none port=return

	IMAGE_C3 img_0(WIDTH, WIDTH);
    IMAGE_C1 tmp_1(WIDTH, WIDTH);
    IMAGE_C1 tmp_2(WIDTH, WIDTH);
    IMAGE_C1 tmp_3(WIDTH, WIDTH);
	IMAGE_C1 tmp_4(WIDTH, WIDTH);
    IMAGE_C1 tmp_5(WIDTH, WIDTH);
	IMAGE_C1 tmp_6(WIDTH, WIDTH);
    IMAGE_C1 tmp_7(WIDTH, WIDTH);

	data valuesImage[WIDTH*WIDTH], arrayEqualized[WIDTH*WIDTH], dstArray[WIDTH*WIDTH], hist[N], minIntValue=N;
	data pupThreshold = (data)streamPop<data, UIntAxis, UIntStream>(src_axi);
    
	//Inizialize histogram for equalisation
	for(int i=0; i<256; i++) {
#pragma HLS UNROLL
		hist[i] = 0; }
#pragma HLS dataflow
    
	//Read the stream of data and directly calculate and fill the array containing the image
loop_in: for (int i = 0; i < WIDTH*WIDTH; i++) {

#pragma HLS loop_flatten off
#pragma HLS pipeline II=1
    
    //color conversion
	float channels[CH];
	channels[0] = 0.299*(data)streamPop<unsigned int, UIntAxis, UIntStream>(src_axi);
	channels[1] = 0.587*(data)streamPop<unsigned int, UIntAxis, UIntStream>(src_axi);
	channels[2] = 0.114*(data)streamPop<unsigned int, UIntAxis, UIntStream>(src_axi);

	data value =(data)(channels[0] + channels[1] + channels[2]);
    
	valuesImage[i] = value;
    
	if(value < minIntValue) minIntValue = value;
    //updateing intensity histogram
    hist[value]++;
}

	MyEqualizer(valuesImage, arrayEqualized, hist, minIntValue); //equalization
	hls::Array2Mat<WIDTH, data, WIDTH, WIDTH, HLS_8UC1>(arrayEqualized, tmp_2);//Array to hlsMat object
	hls::Threshold(tmp_2, tmp_3, pupThreshold, 255, 0);//threshold filter
	hls::Erode(tmp_3, tmp_4); //opening operation
	hls::Dilate(tmp_4, tmp_5);
	hls::Dilate(tmp_5, tmp_6);//closing
	hls::Erode(tmp_6, tmp_7);
	hls::Mat2Array<WIDTH, data, WIDTH, WIDTH, HLS_8UC1>(tmp_7, dstArray);

	for(int i=0; i<WIDTH*WIDTH; i++) {
        
#pragma HLS loop_flatten off
#pragma HLS pipeline II=1
        
		streamPush<data, UIntAxis, UIntStream>(dstArray[i], i==(WIDTH*WIDTH-1), dst_axi, 8);
	}
}

void MyEqualizer(data src[], data dst[], data hist[], data cdfMin) {
	ap_uint<12> cdf[N];
	ap_uint<12> sum=0;
	ap_uint<12> factor = (WIDTH*WIDTH -cdfMin);

	for(int i=0;i<N;i++) {
		sum+=hist[i];
		cdf[i]=sum;
    }
loop_height: for (int i = 0; i < WIDTH*WIDTH; i++) {
#pragma HLS PIPELINE
			data index = src[i];
			ap_uint<12> newValue = (int)(255*(cdf[index] - cdfMin)/factor);
			dst[i] = (data)newValue;
	}
}


