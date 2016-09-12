#pragma once


//this method is the base class for all of the classes related to taking Means
template <typename Sub>
struct MeanBase
{
	//provides an average between respective elements in each of the arrays supplied
	//arithmetic mean example with two int arrays for args
	// index    int *     int *   result
	//	0	      1         2       1.5
	//  1         2         8       5
	//arithmetic mean example with two int arrays and a float array for args
	// index    int *    int *    float *  result
	//  0	      1         2       3       2
	//  1         2         8       20      10
	template <typename Result, typename... Targs> //using a variadic template
	__device__
		void CalculateMeans( Result * result, Targs ... args ) const
	{
		const Sub* calculator = static_cast<const Sub*>( this );
		calculator->Calculate( result, args... );
	}
};

struct GeometricMean : MeanBase<GeometricMean>
{
	template <typename Result, typename... Targs>
	__device__ void Calculate( Result * result, Targs ... args ) const
	{
		static_assert( std::is_floating_point<Result>::value, "Only floating points types are allowed for storing calculation results" );
		int i = threadIdx.x;

		result[ i ] = std::powf( recurseiveMult( args... ), 1.f / sizeof...( args ) );
	}

private:
	template <typename T>
	__device__ float recurseiveMult( T* val ) const
	{
		static_assert( std::is_arithmetic<T>::value, "Only arithmetic types supported" );
		return val[ threadIdx.x ];
	}

	template <typename T, typename... Targs>
	__device__ float recurseiveMult( T* val, Targs...args ) const
	{
		static_assert( std::is_arithmetic<T>::value, "Only arithmetic types supported" );
		return static_cast<float>( val[ threadIdx.x ] )*recurseiveMult( args... );
	}
};

struct HarmonicMean : MeanBase<HarmonicMean>
{
	template <typename Result, typename... Targs>
	__device__ void Calculate( Result * result, Targs... args ) const
	{
		static_assert( std::is_floating_point<Result>::value, "Only floating points types are allowed for storing calculation results" );
		int i = threadIdx.x;

		result[ i ] = sizeof...( args ) / recurseiveMult( args... );
	}

private:
	template <typename T>
	__device__ float recurseiveMult( T* val ) const
	{
		static_assert( std::is_arithmetic<T>::value, "Only arithmetic types supported" );
		return ( 1.f / val[ threadIdx.x ] );
	}

	template <typename T, typename... Targs>
	__device__ float recurseiveMult( T* val, Targs...args ) const
	{
		static_assert( std::is_arithmetic<T>::value, "Only arithmetic types supported" );
		return ( 1.f / val[ threadIdx.x ] ) + recurseiveMult( args... );
	}
};
