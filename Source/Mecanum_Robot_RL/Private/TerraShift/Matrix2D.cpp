#include "TerraShift/Matrix2D.h"

// Constructors

Matrix2D::Matrix2D()
    : Rows(0), Columns(0)
{
}

Matrix2D::Matrix2D(int32 InRows, int32 InColumns)
    : Rows(InRows), Columns(InColumns)
{
    Data.SetNum(Rows);
    for (int32 i = 0; i < Rows; ++i)
    {
        Data[i].SetNumZeroed(Columns);
    }
}

Matrix2D::Matrix2D(int32 InRows, int32 InColumns, float InitialValue)
    : Rows(InRows), Columns(InColumns)
{
    Data.SetNum(Rows);
    for (int32 i = 0; i < Rows; ++i)
    {
        Data[i].Init(InitialValue, Columns);
    }
}

Matrix2D::Matrix2D(const TArray<TArray<float>>& InData)
{
    Rows = InData.Num();
    Columns = Rows > 0 ? InData[0].Num() : 0;
    Data = InData;
}

// Copy constructor
Matrix2D::Matrix2D(const Matrix2D& Other)
    : Rows(Other.Rows), Columns(Other.Columns), Data(Other.Data)
{
}

// Assignment operator
Matrix2D& Matrix2D::operator=(const Matrix2D& Other)
{
    if (this != &Other)
    {
        Rows = Other.Rows;
        Columns = Other.Columns;
        Data = Other.Data;
    }
    return *this;
}

// Element-wise operators

Matrix2D Matrix2D::operator+(const Matrix2D& Other) const
{
    check(Rows == Other.Rows && Columns == Other.Columns);

    Matrix2D Result(Rows, Columns);
    for (int32 i = 0; i < Rows; ++i)
    {
        for (int32 j = 0; j < Columns; ++j)
        {
            Result.Data[i][j] = Data[i][j] + Other.Data[i][j];
        }
    }
    return Result;
}

Matrix2D& Matrix2D::operator+=(const Matrix2D& Other)
{
    check(Rows == Other.Rows && Columns == Other.Columns);

    for (int32 i = 0; i < Rows; ++i)
    {
        for (int32 j = 0; j < Columns; ++j)
        {
            Data[i][j] += Other.Data[i][j];
        }
    }
    return *this;
}

Matrix2D Matrix2D::operator-(const Matrix2D& Other) const
{
    check(Rows == Other.Rows && Columns == Other.Columns);

    Matrix2D Result(Rows, Columns);
    for (int32 i = 0; i < Rows; ++i)
    {
        for (int32 j = 0; j < Columns; ++j)
        {
            Result.Data[i][j] = Data[i][j] - Other.Data[i][j];
        }
    }
    return Result;
}

Matrix2D& Matrix2D::operator-=(const Matrix2D& Other)
{
    check(Rows == Other.Rows && Columns == Other.Columns);

    for (int32 i = 0; i < Rows; ++i)
    {
        for (int32 j = 0; j < Columns; ++j)
        {
            Data[i][j] -= Other.Data[i][j];
        }
    }
    return *this;
}

Matrix2D Matrix2D::operator*(const Matrix2D& Other) const
{
    check(Rows == Other.Rows && Columns == Other.Columns);

    Matrix2D Result(Rows, Columns);
    for (int32 i = 0; i < Rows; ++i)
    {
        for (int32 j = 0; j < Columns; ++j)
        {
            Result.Data[i][j] = Data[i][j] * Other.Data[i][j];
        }
    }
    return Result;
}

Matrix2D& Matrix2D::operator*=(const Matrix2D& Other)
{
    check(Rows == Other.Rows && Columns == Other.Columns);

    for (int32 i = 0; i < Rows; ++i)
    {
        for (int32 j = 0; j < Columns; ++j)
        {
            Data[i][j] *= Other.Data[i][j];
        }
    }
    return *this;
}

Matrix2D Matrix2D::operator/(const Matrix2D& Other) const
{
    check(Rows == Other.Rows && Columns == Other.Columns);

    Matrix2D Result(Rows, Columns);
    for (int32 i = 0; i < Rows; ++i)
    {
        for (int32 j = 0; j < Columns; ++j)
        {
            check(Other.Data[i][j] != 0.0f);
            Result.Data[i][j] = Data[i][j] / Other.Data[i][j];
        }
    }
    return Result;
}

Matrix2D& Matrix2D::operator/=(const Matrix2D& Other)
{
    check(Rows == Other.Rows && Columns == Other.Columns);

    for (int32 i = 0; i < Rows; ++i)
    {
        for (int32 j = 0; j < Columns; ++j)
        {
            check(Other.Data[i][j] != 0.0f);
            Data[i][j] /= Other.Data[i][j];
        }
    }
    return *this;
}

// Scalar operations

Matrix2D Matrix2D::operator+(float Scalar) const
{
    Matrix2D Result(Rows, Columns);
    for (int32 i = 0; i < Rows; ++i)
    {
        for (int32 j = 0; j < Columns; ++j)
        {
            Result.Data[i][j] = Data[i][j] + Scalar;
        }
    }
    return Result;
}

Matrix2D& Matrix2D::operator+=(float Scalar)
{
    for (int32 i = 0; i < Rows; ++i)
    {
        for (int32 j = 0; j < Columns; ++j)
        {
            Data[i][j] += Scalar;
        }
    }
    return *this;
}

Matrix2D Matrix2D::operator-(float Scalar) const
{
    Matrix2D Result(Rows, Columns);
    for (int32 i = 0; i < Rows; ++i)
    {
        for (int32 j = 0; j < Columns; ++j)
        {
            Result.Data[i][j] = Data[i][j] - Scalar;
        }
    }
    return Result;
}

Matrix2D& Matrix2D::operator-=(float Scalar)
{
    for (int32 i = 0; i < Rows; ++i)
    {
        for (int32 j = 0; j < Columns; ++j)
        {
            Data[i][j] -= Scalar;
        }
    }
    return *this;
}

Matrix2D Matrix2D::operator*(float Scalar) const
{
    Matrix2D Result(Rows, Columns);
    for (int32 i = 0; i < Rows; ++i)
    {
        for (int32 j = 0; j < Columns; ++j)
        {
            Result.Data[i][j] = Data[i][j] * Scalar;
        }
    }
    return Result;
}

Matrix2D& Matrix2D::operator*=(float Scalar)
{
    for (int32 i = 0; i < Rows; ++i)
    {
        for (int32 j = 0; j < Columns; ++j)
        {
            Data[i][j] *= Scalar;
        }
    }
    return *this;
}

Matrix2D Matrix2D::operator/(float Scalar) const
{
    check(Scalar != 0.0f);
    Matrix2D Result(Rows, Columns);
    float InvScalar = 1.0f / Scalar;
    for (int32 i = 0; i < Rows; ++i)
    {
        for (int32 j = 0; j < Columns; ++j)
        {
            Result.Data[i][j] = Data[i][j] * InvScalar;
        }
    }
    return Result;
}

Matrix2D& Matrix2D::operator/=(float Scalar)
{
    check(Scalar != 0.0f);
    float InvScalar = 1.0f / Scalar;
    for (int32 i = 0; i < Rows; ++i)
    {
        for (int32 j = 0; j < Columns; ++j)
        {
            Data[i][j] *= InvScalar;
        }
    }
    return *this;
}

// Scalar operations with scalar on the left
Matrix2D operator+(float Scalar, const Matrix2D& Matrix)
{
    return Matrix + Scalar;
}

Matrix2D operator-(float Scalar, const Matrix2D& Matrix)
{
    Matrix2D Result(Matrix.Rows, Matrix.Columns);
    for (int32 i = 0; i < Matrix.Rows; ++i)
    {
        for (int32 j = 0; j < Matrix.Columns; ++j)
        {
            Result[i][j] = Scalar - Matrix.Data[i][j];
        }
    }
    return Result;
}

Matrix2D operator*(float Scalar, const Matrix2D& Matrix)
{
    return Matrix * Scalar;
}

Matrix2D operator/(float Scalar, const Matrix2D& Matrix)
{
    Matrix2D Result(Matrix.Rows, Matrix.Columns);
    for (int32 i = 0; i < Matrix.Rows; ++i)
    {
        for (int32 j = 0; j < Matrix.Columns; ++j)
        {
            check(Matrix.Data[i][j] != 0.0f);
            Result[i][j] = Scalar / Matrix.Data[i][j];
        }
    }
    return Result;
}

// Element access

TArray<float>& Matrix2D::operator[](int32 RowIndex)
{
    // Support negative indexing
    if (RowIndex < 0)
    {
        RowIndex = Rows + RowIndex;
    }
    check(RowIndex >= 0 && RowIndex < Rows);
    return Data[RowIndex];
}

const TArray<float>& Matrix2D::operator[](int32 RowIndex) const
{
    // Support negative indexing
    if (RowIndex < 0)
    {
        RowIndex = Rows + RowIndex;
    }
    check(RowIndex >= 0 && RowIndex < Rows);
    return Data[RowIndex];
}

// Sub-matrix extraction

Matrix2D Matrix2D::Sub(int32 RowStart, int32 RowEnd, int32 ColStart, int32 ColEnd) const
{
    // Adjust indices if negative
    if (RowStart < 0) RowStart = Rows + RowStart;
    if (RowEnd < 0) RowEnd = Rows + RowEnd;
    if (ColStart < 0) ColStart = Columns + ColStart;
    if (ColEnd < 0) ColEnd = Columns + ColEnd;

    check(RowStart >= 0 && RowStart < Rows);
    check(RowEnd >= 0 && RowEnd < Rows);
    check(ColStart >= 0 && ColStart < Columns);
    check(ColEnd >= 0 && ColEnd < Columns);
    check(RowStart <= RowEnd && ColStart <= ColEnd);

    int32 NewRows = RowEnd - RowStart + 1;
    int32 NewCols = ColEnd - ColStart + 1;

    Matrix2D Result(NewRows, NewCols);
    for (int32 i = 0; i < NewRows; ++i)
    {
        for (int32 j = 0; j < NewCols; ++j)
        {
            Result.Data[i][j] = Data[RowStart + i][ColStart + j];
        }
    }
    return Result;
}

// Mathematical functions

Matrix2D Matrix2D::Exp() const
{
    Matrix2D Result(Rows, Columns);
    for (int32 i = 0; i < Rows; ++i)
    {
        Result.Data[i].SetNumUninitialized(Columns);
        for (int32 j = 0; j < Columns; ++j)
        {
            Result.Data[i][j] = FMath::Exp(Data[i][j]);
        }
    }
    return Result;
}

Matrix2D Matrix2D::Cos() const
{
    Matrix2D Result(Rows, Columns);
    for (int32 i = 0; i < Rows; ++i)
    {
        Result.Data[i].SetNumUninitialized(Columns);
        for (int32 j = 0; j < Columns; ++j)
        {
            Result.Data[i][j] = FMath::Cos(Data[i][j]);
        }
    }
    return Result;
}

Matrix2D Matrix2D::Sin() const
{
    Matrix2D Result(Rows, Columns);
    for (int32 i = 0; i < Rows; ++i)
    {
        Result.Data[i].SetNumUninitialized(Columns);
        for (int32 j = 0; j < Columns; ++j)
        {
            Result.Data[i][j] = FMath::Sin(Data[i][j]);
        }
    }
    return Result;
}

Matrix2D Matrix2D::Tanh() const
{
    Matrix2D Result(Rows, Columns);
    for (int32 i = 0; i < Rows; ++i)
    {
        Result.Data[i].SetNumUninitialized(Columns);
        for (int32 j = 0; j < Columns; ++j)
        {
            Result.Data[i][j] = FMath::Tanh(Data[i][j]);
        }
    }
    return Result;
}

// Matrix functions

float Matrix2D::Dot(const Matrix2D& Other) const
{
    check(Rows == Other.Rows && Columns == Other.Columns);

    float Sum = 0.0f;
    for (int32 i = 0; i < Rows; ++i)
    {
        for (int32 j = 0; j < Columns; ++j)
        {
            Sum += Data[i][j] * Other.Data[i][j];
        }
    }
    return Sum;
}

float Matrix2D::Norm() const
{
    float SumSquares = 0.0f;
    for (const auto& Row : Data)
    {
        for (float Value : Row)
        {
            SumSquares += Value * Value;
        }
    }
    return FMath::Sqrt(SumSquares);
}

void Matrix2D::Clip(float MinValue, float MaxValue)
{
    for (auto& Row : Data)
    {
        for (float& Value : Row)
        {
            Value = FMath::Clamp(Value, MinValue, MaxValue);
        }
    }
}

void Matrix2D::Init(float Value)
{
    for (auto& Row : Data)
    {
        Row.Init(Value, Columns);
    }
}

// Utility functions

FString Matrix2D::ToString() const
{
    FString Result;
    for (const auto& Row : Data)
    {
        for (float Value : Row)
        {
            Result += FString::Printf(TEXT("%6.2f "), Value);
        }
        Result += TEXT("\n");
    }
    return Result;
}

// Get dimensions

int32 Matrix2D::GetNumRows() const
{
    return Rows;
}

int32 Matrix2D::GetNumColumns() const
{
    return Columns;
}

// Get the minimum value in the matrix
float Matrix2D::Min() const
{
    check(Rows > 0 && Columns > 0);

    float MinValue = Data[0][0];
    for (int32 i = 0; i < Rows; ++i)
    {
        for (int32 j = 0; j < Columns; ++j)
        {
            if (Data[i][j] < MinValue)
            {
                MinValue = Data[i][j];
            }
        }
    }
    return MinValue;
}

// Get the maximum value in the matrix
float Matrix2D::Max() const
{
    check(Rows > 0 && Columns > 0);

    float MaxValue = Data[0][0];
    for (int32 i = 0; i < Rows; ++i)
    {
        for (int32 j = 0; j < Columns; ++j)
        {
            if (Data[i][j] > MaxValue)
            {
                MaxValue = Data[i][j];
            }
        }
    }
    return MaxValue;
}
