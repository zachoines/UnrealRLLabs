#pragma once
#include "Math/UnrealMathUtility.h"
#include "CoreMinimal.h"

class UNREALRLLABS_API Matrix2D
{
private:
    TArray<TArray<float>> Data;
    int32 Rows;
    int32 Columns;

public:
    // Constructors
    Matrix2D();
    Matrix2D(int32 InRows, int32 InColumns);
    Matrix2D(int32 InRows, int32 InColumns, float InitialValue);
    Matrix2D(const TArray<TArray<float>>& InData);

    // Copy constructor and assignment operator
    Matrix2D(const Matrix2D& Other);
    Matrix2D& operator=(const Matrix2D& Other);

    // Element-wise operators
    Matrix2D operator+(const Matrix2D& Other) const;
    Matrix2D& operator+=(const Matrix2D& Other);
    Matrix2D operator-(const Matrix2D& Other) const;
    Matrix2D& operator-=(const Matrix2D& Other);
    Matrix2D operator*(const Matrix2D& Other) const;
    Matrix2D& operator*=(const Matrix2D& Other);
    Matrix2D operator/(const Matrix2D& Other) const;
    Matrix2D& operator/=(const Matrix2D& Other);

    // Scalar operations
    Matrix2D operator+(float Scalar) const;
    Matrix2D& operator+=(float Scalar);
    Matrix2D operator-(float Scalar) const;
    Matrix2D& operator-=(float Scalar);
    Matrix2D operator*(float Scalar) const;
    Matrix2D& operator*=(float Scalar);
    Matrix2D operator/(float Scalar) const;
    Matrix2D& operator/=(float Scalar);

    // Element access
    TArray<float>& operator[](int32 RowIndex);
    const TArray<float>& operator[](int32 RowIndex) const;

    // Sub-matrix extraction
    Matrix2D Sub(int32 RowStart, int32 RowEnd, int32 ColStart, int32 ColEnd) const;

    // Mathematical functions
    Matrix2D Exp() const;
    Matrix2D Cos() const;
    Matrix2D Sin() const;
    Matrix2D Tanh() const;

    // Matrix functions
    float Dot(const Matrix2D& Other) const;
    float Norm() const;
    void Clip(float MinValue, float MaxValue);
    void Init(float Value);

    // Utility functions
    FString ToString() const;

    // Get dimensions
    int32 GetNumRows() const;
    int32 GetNumColumns() const;

    // Additional operators for scalar operations
    friend Matrix2D operator+(float Scalar, const Matrix2D& Matrix);
    friend Matrix2D operator-(float Scalar, const Matrix2D& Matrix);
    friend Matrix2D operator*(float Scalar, const Matrix2D& Matrix);
    friend Matrix2D operator/(float Scalar, const Matrix2D& Matrix);
    
    float Min() const;
    float Max() const;
};