
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <limits>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>
#include "DeviceCode.h"
#include <iostream>


#define SIZE 10



template <typename Result, typename... Targs>
__global__
void TestGeometricMean( Result * result, Targs ... args )
{
	GeometricMean().CalculateMeans( result, args... );
}

template <typename Result, typename... Targs>
__global__
void TestHarmonicMean( Result * result, Targs ... args )
{
	HarmonicMean().CalculateMeans( result, args... );
}

template<typename T>
void PrintEntries( const char* label, T* entries )
{
	std::cout << label << std::endl;
	for ( auto i = 0; i < SIZE; ++i )
	{
		std::cout << i << "\t-\t" << entries[ i ] << std::endl;
	}
}

int main()
{
	int *setOfInts1, *setOfInts2;
	float * setOfFloats;
	double * setOfDoubles;

	//allocate and initialize the various inputs
	{
		setOfInts1 = (int *)malloc( SIZE * sizeof( int ) );
		setOfInts2 = (int *)malloc( SIZE * sizeof( int ) );
		setOfFloats = (float *)malloc( SIZE * sizeof( float ) );
		setOfDoubles = (double *)malloc( SIZE * sizeof( double ) );
		
		int temp;
		for ( int i = 0; i < SIZE; ++i )
		{
			temp = i + 1;//we don't really care what our values but we do care that none are 0 since Harmonic and Geometric means don't like 0

			setOfInts1[ i ] = temp;
			setOfInts2[ i ] = temp * 2;
			setOfFloats[ i ] = temp*1.3f;
			setOfDoubles[ i ] = temp*2.3;
		}
	}

	int * d_setOfInts1, *d_setOfInts2;
	float * d_setOfFloats;
	double * d_setOfDoubles;
	//allocate and initialize the various device side versions of the inputs
	{
		cudaMalloc( &d_setOfInts1, SIZE * sizeof( int ) );
		cudaMalloc( &d_setOfInts2, SIZE * sizeof( int ) );
		cudaMalloc( &d_setOfFloats, SIZE * sizeof( float ) );
		cudaMalloc( &d_setOfDoubles, SIZE * sizeof( double ) );

		cudaMemcpy( d_setOfInts1, setOfInts1, SIZE * sizeof( int ), cudaMemcpyHostToDevice );
		cudaMemcpy( d_setOfInts2, setOfInts2, SIZE * sizeof( int ), cudaMemcpyHostToDevice );
		cudaMemcpy( d_setOfFloats, setOfFloats, SIZE * sizeof( float ), cudaMemcpyHostToDevice );
		cudaMemcpy( d_setOfDoubles, setOfDoubles, SIZE * sizeof( double ), cudaMemcpyHostToDevice );
	}


	float* resultFloat = (float *)malloc( SIZE * sizeof( float ) );
	double* resultDouble = (double *)malloc( SIZE * sizeof( double ) );

	float * d_resultFloat;
	double * d_resultDouble;
	cudaMalloc( &d_resultFloat, SIZE * sizeof( float ) );
	cudaMalloc( &d_resultDouble, SIZE * sizeof( double ) );


	TestHarmonicMean << <1, SIZE >> > ( d_resultFloat, d_setOfInts1, d_setOfInts2 );
	{
		cudaError_t errorCode = cudaMemcpy( resultFloat, d_resultFloat, SIZE * sizeof( float ), cudaMemcpyDeviceToHost );
		PrintEntries( "Harmonic Mean of some ints with some other ints stored in floats:", resultFloat );
	}

	TestHarmonicMean << <1, SIZE >> > ( d_resultFloat, d_setOfInts1, d_setOfFloats );
	{
		cudaError_t errorCode = cudaMemcpy( resultFloat, d_resultFloat, SIZE * sizeof( float ), cudaMemcpyDeviceToHost );
		PrintEntries( "Harmonic Mean of some ints with some floats stored in floats:", resultFloat );
	}

	TestHarmonicMean << <1, SIZE >> > ( d_resultDouble, d_setOfInts1, d_setOfFloats );
	{
		cudaError_t errorCode = cudaMemcpy( resultDouble, d_resultDouble, SIZE * sizeof( double ), cudaMemcpyDeviceToHost );
		PrintEntries( "Harmonic Mean of some ints with some floats stored in doubles:", resultDouble );
	}

	TestHarmonicMean << <1, SIZE >> > ( d_resultDouble, d_setOfDoubles, d_setOfFloats, d_setOfInts1 );
	{
		cudaError_t errorCode = cudaMemcpy( resultDouble, d_resultDouble, SIZE * sizeof( double ), cudaMemcpyDeviceToHost );
		PrintEntries( "Harmonic Mean of some doubles, floats, and ints stored in doubles:", resultDouble );
	}


	TestGeometricMean << <1, SIZE >> > ( d_resultFloat, d_setOfInts1, d_setOfInts1, d_setOfInts1, d_setOfInts1, d_setOfInts2 );
	{
		cudaError_t errorCode = cudaMemcpy( resultFloat, d_resultFloat, SIZE * sizeof( float ), cudaMemcpyDeviceToHost );
		PrintEntries( "Geometric Mean of the same set of ints 4 times and some other ints stored in floats:", resultFloat );
	}
	TestGeometricMean << <1, SIZE >> > ( d_resultDouble, d_setOfInts1, d_setOfFloats, d_setOfDoubles );
	{
		cudaError_t errorCode = cudaMemcpy( resultDouble, d_resultDouble, SIZE * sizeof( double ), cudaMemcpyDeviceToHost );
		PrintEntries( "Harmonic Mean of some ints, floats, and doubles stored in doubles:", resultDouble );
	}
	TestGeometricMean << <1, SIZE >> > ( d_resultDouble, d_setOfDoubles, d_setOfFloats );
	{
		cudaError_t errorCode = cudaMemcpy( resultDouble, d_resultDouble, SIZE * sizeof( double ), cudaMemcpyDeviceToHost );
		PrintEntries( "Harmonic Mean of some doubles and floats stored in doubles:", resultDouble );
	}

	cudaFree( d_resultDouble );
	cudaFree( d_resultFloat );
	cudaFree( d_setOfInts1 );
	cudaFree( d_setOfInts2 );
	cudaFree( d_setOfFloats );
	cudaFree( d_setOfDoubles );

	free( resultFloat );
	free( resultDouble );
	free( setOfInts1 );
	free( setOfInts2 );
	free( setOfFloats );
	free( setOfDoubles );

	return 0;
}