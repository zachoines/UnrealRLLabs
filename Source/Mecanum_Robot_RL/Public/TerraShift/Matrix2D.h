#pragma once

#include "CoreMinimal.h"
#include "Math/UnrealMathUtility.h"
#include "Matrix2D.generated.h"

/**
 * A "row proxy" that allows user-friendly [row][col] access while
 * internally referencing a single TArray<float> storage.
 */
struct FFMatrix2DRowProxy
{
    // Pointer to the parent FMatrix2D's data
    float* RowData;
    // Number of columns in the matrix
    int32 NumCols;

    // Indexing operator gives element reference at col
    float& operator[](int32 ColIndex)
    {
        check(ColIndex >= 0 && ColIndex < NumCols);
        return RowData[ColIndex];
    }

    const float& operator[](int32 ColIndex) const
    {
        check(ColIndex >= 0 && ColIndex < NumCols);
        return RowData[ColIndex];
    }
};

/**
 * A simple 2D matrix struct for storing and operating on float data,
 * implemented via a single TArray<float> to avoid nested TArray issues.
 */
USTRUCT(BlueprintType)
struct UNREALRLLABS_API FMatrix2D
{
    GENERATED_BODY()

private:
    // Single TArray of size (Rows * Columns) for data
    UPROPERTY()
    TArray<float> Data;

    UPROPERTY()
    int32 Rows;

    UPROPERTY()
    int32 Columns;

public:
    // Constructors
    FMatrix2D();
    FMatrix2D(int32 InRows, int32 InColumns);
    FMatrix2D(int32 InRows, int32 InColumns, float InitialValue);
    FMatrix2D(const TArray<TArray<float>>& InData);

    // Copy constructor and assignment operator
    FMatrix2D(const FMatrix2D& Other);
    FMatrix2D& operator=(const FMatrix2D& Other);

    // Element-wise operators
    FMatrix2D operator+(const FMatrix2D& Other) const;
    FMatrix2D& operator+=(const FMatrix2D& Other);
    FMatrix2D operator-(const FMatrix2D& Other) const;
    FMatrix2D& operator-=(const FMatrix2D& Other);
    FMatrix2D operator*(const FMatrix2D& Other) const;
    FMatrix2D& operator*=(const FMatrix2D& Other);
    FMatrix2D operator/(const FMatrix2D& Other) const;
    FMatrix2D& operator/=(const FMatrix2D& Other);

    // Scalar operations
    FMatrix2D operator+(float Scalar) const;
    FMatrix2D& operator+=(float Scalar);
    FMatrix2D operator-(float Scalar) const;
    FMatrix2D& operator-=(float Scalar);
    FMatrix2D operator*(float Scalar) const;
    FMatrix2D& operator*=(float Scalar);
    FMatrix2D operator/(float Scalar) const;
    FMatrix2D& operator/=(float Scalar);

    // **Row indexing**: returns a proxy object so user can do matrix[row][col].
    FFMatrix2DRowProxy operator[](int32 RowIndex);
    const FFMatrix2DRowProxy operator[](int32 RowIndex) const;

    // Sub-matrix extraction
    FMatrix2D Sub(int32 RowStart, int32 RowEnd, int32 ColStart, int32 ColEnd) const;

    // Mathematical functions
    FMatrix2D Exp() const;
    FMatrix2D Cos() const;
    FMatrix2D Sin() const;
    FMatrix2D Tanh() const;

    // Matrix functions
    float Dot(const FMatrix2D& Other) const;
    float Norm() const;
    void Clip(float MinValue, float MaxValue);
    void Init(float Value);

    // Utility functions
    FString ToString() const;

    // Get dimensions
    int32 GetNumRows() const;
    int32 GetNumColumns() const;

    // Additional operators for scalar operations
    friend FMatrix2D operator+(float Scalar, const FMatrix2D& Matrix);
    friend FMatrix2D operator-(float Scalar, const FMatrix2D& Matrix);
    friend FMatrix2D operator*(float Scalar, const FMatrix2D& Matrix);
    friend FMatrix2D operator/(float Scalar, const FMatrix2D& Matrix);

    // Min/Max
    float Min() const;
    float Max() const;

    // New Helper Methods

    /** Returns a FMatrix2D whose elements are the absolute values of this matrix. */
    FMatrix2D Abs() const;

    /** Returns the sum of all elements in this matrix. */
    float Sum() const;

    /** Returns the arithmetic mean of all elements in this matrix. */
    float Mean() const;

    // Transpose the matrix
    FMatrix2D T() const;

    // Standard matrix multiplication
    FMatrix2D MatMul(const FMatrix2D& Other) const;

private:
    /** Helper to get the index in the single array for (row,col). */
    FORCEINLINE int32 LinearIndex(int32 RowIndex, int32 ColIndex) const
    {
        return (RowIndex * Columns) + ColIndex;
    }
};