/*
 * Copyright 2016 Gianluca Durelli
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 * 	http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#ifndef NECSTREAM
#define NECSTREAM

#define AP_INT_MAX_W 4096
#include "ap_axi_sdata.h"
#include <hls_stream.h>
#include <hls_video.h>

#define BIT_WIDTH 32
#include <ap_fixed.h>

typedef ap_fixed<48, 32, AP_TRN> data_t;

template<class DT, int D,int U,int TI,int TD>
  struct my_ap_axis{
    DT       data;
    ap_uint<(D+7)/8> keep;
    ap_uint<(D+7)/8> strb;
    ap_uint<U>       user;
    ap_uint<1>       last;
    ap_uint<TI>      id;
    ap_uint<TD>      dest;
  };

template <class AXI>
void memsetAXIS_Data(AXI & d, unsigned int bitWidth){
	d.data = 0;
	d.last = 0;
	d.keep = ( 1<<(bitWidth/8) ) - 1; // Keep all the bytes
	d.strb = ( 1<<(bitWidth/8) ) - 1; // All the bytes are data

	// Set to default value if not using
	d.dest = 0;
	d.id = 0;
	d.user = 0;
}

typedef my_ap_axis<float,32,1,1,1> FloatAxis;
typedef my_ap_axis<ap_uint<8>,8,1,1,1> UIntAxis;
typedef hls::stream<FloatAxis> FloatStream;
typedef hls::stream<UIntAxis> UIntStream;

template <class RET, class DATA, class STREAM>
RET streamPop(STREAM &stream){
	RET value;
	DATA axisData;

	stream >> axisData;
	value = axisData.data;

	return value;
}

template <class TYPE, class DATA, class STREAM>
void streamPush(TYPE value, int last, STREAM &stream, int bitWidth){
	DATA d;
	memsetAXIS_Data<DATA>(d, bitWidth);
	d.data = value;
	d.last = last;
	stream << d;
}

template <class TYPE, class DATA, class STREAM>
void streamForward(STREAM &streamIn, STREAM &streamOut, unsigned int items, unsigned int bitWidth){
	int c;
	for(c=0; c<items; c++){
		TYPE tmp = streamPop<TYPE, DATA, STREAM>(streamIn);
		streamPush<TYPE, DATA, STREAM>(tmp, 0, streamOut, bitWidth);
	}
}

template <class TYPE, class DATA, class STREAM>
void readBuffer(TYPE *buffer, STREAM &stream, unsigned int items){
	for(int k=0; k<items; k++){
		buffer[k] = streamPop<TYPE, DATA, STREAM>(stream);
	}
}

template <class TYPE, class DATA, class STREAM>
void sendBuffer(TYPE *buffer, STREAM &stream, unsigned int items, unsigned int bitWidth){
	for(int k=0; k<items; k++){
		streamPush<TYPE, DATA, STREAM>(buffer[k], 0, stream, bitWidth);
	}
}

#endif
