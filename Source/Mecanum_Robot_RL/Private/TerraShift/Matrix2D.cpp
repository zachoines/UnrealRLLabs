#include "TerraShift/Matrix2D.h"

// Constructors
FMatrix2D::FMatrix2D()
    : Rows(0)
    , Columns(0)
{
}

FMatrix2D::FMatrix2D(int32 InRows, int32 InColumns)
    : Rows(InRows)
    , Columns(InColumns)
{
    Data.SetNum(Rows * Columns);
}

FMatrix2D::FMatrix2D(int32 InRows, int32 InColumns, float InitialValue)
    : Rows(InRows)
    , Columns(InColumns)
{
    Data.SetNum(Rows * Columns);
    for (int32 i = 0; i < Data.Num(); ++i)
    {
        Data[i] = InitialValue;
    }
}

FMatrix2D::FMatrix2D(const TArray<TArray<float>>& InData)
{
    Rows = InData.Num();
    Columns = (Rows > 0) ? InData[0].Num() : 0;
    Data.SetNum(Rows * Columns);

    for (int32 r = 0; r < Rows; ++r)
    {
        check(InData[r].Num() == Columns);  // ensure rectangular
        for (int32 c = 0; c < Columns; ++c)
        {
            Data[LinearIndex(r, c)] = InData[r][c];
        }
    }
}

// Copy constructor
FMatrix2D::FMatrix2D(const FMatrix2D& Other)
    : Rows(Other.Rows)
    , Columns(Other.Columns)
    , Data(Other.Data)
{
}

// Assignment operator
FMatrix2D& FMatrix2D::operator=(const FMatrix2D& Other)
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
FMatrix2D FMatrix2D::operator+(const FMatrix2D& Other) const
{
    check(Rows == Other.Rows && Columns == Other.Columns);

    FMatrix2D Result(Rows, Columns);
    for (int32 i = 0; i < Data.Num(); ++i)
    {
        Result.Data[i] = Data[i] + Other.Data[i];
    }
    return Result;
}

FMatrix2D& FMatrix2D::operator+=(const FMatrix2D& Other)
{
    check(Rows == Other.Rows && Columns == Other.Columns);

    for (int32 i = 0; i < Data.Num(); ++i)
    {
        Data[i] += Other.Data[i];
    }
    return *this;
}

FMatrix2D FMatrix2D::operator-(const FMatrix2D& Other) const
{
    check(Rows == Other.Rows && Columns == Other.Columns);

    FMatrix2D Result(Rows, Columns);
    for (int32 i = 0; i < Data.Num(); ++i)
    {
        Result.Data[i] = Data[i] - Other.Data[i];
    }
    return Result;
}

FMatrix2D& FMatrix2D::operator-=(const FMatrix2D& Other)
{
    check(Rows == Other.Rows && Columns == Other.Columns);

    for (int32 i = 0; i < Data.Num(); ++i)
    {
        Data[i] -= Other.Data[i];
    }
    return *this;
}

FMatrix2D FMatrix2D::operator*(const FMatrix2D& Other) const
{
    check(Rows == Other.Rows && Columns == Other.Columns);

    FMatrix2D Result(Rows, Columns);
    for (int32 i = 0; i < Data.Num(); ++i)
    {
        Result.Data[i] = Data[i] * Other.Data[i];
    }
    return Result;
}

FMatrix2D& FMatrix2D::operator*=(const FMatrix2D& Other)
{
    check(Rows == Other.Rows && Columns == Other.Columns);

    for (int32 i = 0; i < Data.Num(); ++i)
    {
        Data[i] *= Other.Data[i];
    }
    return *this;
}

FMatrix2D FMatrix2D::operator/(const FMatrix2D& Other) const
{
    check(Rows == Other.Rows && Columns == Other.Columns);

    FMatrix2D Result(Rows, Columns);
    for (int32 i = 0; i < Data.Num(); ++i)
    {
        check(Other.Data[i] != 0.0f);
        Result.Data[i] = Data[i] / Other.Data[i];
    }
    return Result;
}

FMatrix2D& FMatrix2D::operator/=(const FMatrix2D& Other)
{
    check(Rows == Other.Rows && Columns == Other.Columns);

    for (int32 i = 0; i < Data.Num(); ++i)
    {
        check(Other.Data[i] != 0.0f);
        Data[i] /= Other.Data[i];
    }
    return *this;
}

// Scalar operations
FMatrix2D FMatrix2D::operator+(float Scalar) const
{
    FMatrix2D Result(Rows, Columns);
    for (int32 i = 0; i < Data.Num(); ++i)
    {
        Result.Data[i] = Data[i] + Scalar;
    }
    return Result;
}

FMatrix2D& FMatrix2D::operator+=(float Scalar)
{
    for (int32 i = 0; i < Data.Num(); ++i)
    {
        Data[i] += Scalar;
    }
    return *this;
}

FMatrix2D FMatrix2D::operator-(float Scalar) const
{
    FMatrix2D Result(Rows, Columns);
    for (int32 i = 0; i < Data.Num(); ++i)
    {
        Result.Data[i] = Data[i] - Scalar;
    }
    return Result;
}

FMatrix2D& FMatrix2D::operator-=(float Scalar)
{
    for (int32 i = 0; i < Data.Num(); ++i)
    {
        Data[i] -= Scalar;
    }
    return *this;
}

FMatrix2D FMatrix2D::operator*(float Scalar) const
{
    FMatrix2D Result(Rows, Columns);
    for (int32 i = 0; i < Data.Num(); ++i)
    {
        Result.Data[i] = Data[i] * Scalar;
    }
    return Result;
}

FMatrix2D& FMatrix2D::operator*=(float Scalar)
{
    for (int32 i = 0; i < Data.Num(); ++i)
    {
        Data[i] *= Scalar;
    }
    return *this;
}

FMatrix2D FMatrix2D::operator/(float Scalar) const
{
    check(Scalar != 0.0f);
    FMatrix2D Result(Rows, Columns);
    float InvScalar = 1.0f / Scalar;
    for (int32 i = 0; i < Data.Num(); ++i)
    {
        Result.Data[i] = Data[i] * InvScalar;
    }
    return Result;
}

FMatrix2D& FMatrix2D::operator/=(float Scalar)
{
    check(Scalar != 0.0f);
    float InvScalar = 1.0f / Scalar;
    for (int32 i = 0; i < Data.Num(); ++i)
    {
        Data[i] *= InvScalar;
    }
    return *this;
}

// Scalar operations with scalar on the left
FMatrix2D operator+(float Scalar, const FMatrix2D& Matrix)
{
    return Matrix + Scalar;
}

FMatrix2D operator-(float Scalar, const FMatrix2D& Matrix)
{
    FMatrix2D Result(Matrix.Rows, Matrix.Columns);
    for (int32 i = 0; i < Matrix.Data.Num(); ++i)
    {
        Result.Data[i] = Scalar - Matrix.Data[i];
    }
    return Result;
}

FMatrix2D operator*(float Scalar, const FMatrix2D& Matrix)
{
    return Matrix * Scalar;
}

FMatrix2D operator/(float Scalar, const FMatrix2D& Matrix)
{
    FMatrix2D Result(Matrix.Rows, Matrix.Columns);
    for (int32 i = 0; i < Matrix.Data.Num(); ++i)
    {
        check(Matrix.Data[i] != 0.0f);
        Result.Data[i] = Scalar / Matrix.Data[i];
    }
    return Result;
}

// Row indexing: returns a proxy object so user can do M[row][col].
FFMatrix2DRowProxy FMatrix2D::operator[](int32 RowIndex)
{
    // Negative index support
    if (RowIndex < 0)
    {
        RowIndex = Rows + RowIndex;
    }
    check(RowIndex >= 0 && RowIndex < Rows);

    FFMatrix2DRowProxy Proxy;
    Proxy.RowData = &Data[RowIndex * Columns];
    Proxy.NumCols = Columns;
    return Proxy;
}

const FFMatrix2DRowProxy FMatrix2D::operator[](int32 RowIndex) const
{
    // Negative index support
    if (RowIndex < 0)
    {
        RowIndex = Rows + RowIndex;
    }
    check(RowIndex >= 0 && RowIndex < Rows);

    FFMatrix2DRowProxy Proxy;
    // We need const-unsafe cast or separate approach,
    // but for simplicity we do this (const float*) -> (float*)
    // There's no direct built-in const TArray cast, so we do a hack:
    float* RowPtr = const_cast<float*>(&Data[RowIndex * Columns]);
    Proxy.RowData = RowPtr;
    Proxy.NumCols = Columns;
    return Proxy;
}

// Sub-matrix extraction
FMatrix2D FMatrix2D::Sub(int32 RowStart, int32 RowEnd, int32 ColStart, int32 ColEnd) const
{
    // Adjust indices if negative
    if (RowStart < 0) RowStart = Rows + RowStart;
    if (RowEnd < 0)   RowEnd = Rows + RowEnd;
    if (ColStart < 0) ColStart = Columns + ColStart;
    if (ColEnd < 0)   ColEnd = Columns + ColEnd;

    check(RowStart >= 0 && RowStart < Rows);
    check(RowEnd >= 0 && RowEnd < Rows);
    check(ColStart >= 0 && ColStart < Columns);
    check(ColEnd >= 0 && ColEnd < Columns);
    check(RowStart <= RowEnd && ColStart <= ColEnd);

    int32 NewRows = RowEnd - RowStart + 1;
    int32 NewCols = ColEnd - ColStart + 1;

    FMatrix2D Result(NewRows, NewCols);
    for (int32 r = 0; r < NewRows; ++r)
    {
        for (int32 c = 0; c < NewCols; ++c)
        {
            int32 OldIndex = (RowStart + r) * Columns + (ColStart + c);
            Result.Data[r * NewCols + c] = Data[OldIndex];
        }
    }
    return Result;
}

// Math functions
FMatrix2D FMatrix2D::Exp() const
{
    FMatrix2D Result(Rows, Columns);
    for (int32 i = 0; i < Data.Num(); ++i)
    {
        Result.Data[i] = FMath::Exp(Data[i]);
    }
    return Result;
}

FMatrix2D FMatrix2D::Cos() const
{
    FMatrix2D Result(Rows, Columns);
    for (int32 i = 0; i < Data.Num(); ++i)
    {
        Result.Data[i] = FMath::Cos(Data[i]);
    }
    return Result;
}

FMatrix2D FMatrix2D::Sin() const
{
    FMatrix2D Result(Rows, Columns);
    for (int32 i = 0; i < Data.Num(); ++i)
    {
        Result.Data[i] = FMath::Sin(Data[i]);
    }
    return Result;
}

FMatrix2D FMatrix2D::Tanh() const
{
    FMatrix2D Result(Rows, Columns);
    for (int32 i = 0; i < Data.Num(); ++i)
    {
        Result.Data[i] = FMath::Tanh(Data[i]);
    }
    return Result;
}

// Matrix functions
float FMatrix2D::Dot(const FMatrix2D& Other) const
{
    check(Rows == Other.Rows && Columns == Other.Columns);

    float Sum = 0.0f;
    for (int32 i = 0; i < Data.Num(); ++i)
    {
        Sum += Data[i] * Other.Data[i];
    }
    return Sum;
}

float FMatrix2D::Norm() const
{
    float SumSquares = 0.0f;
    for (float Value : Data)
    {
        SumSquares += Value * Value;
    }
    return FMath::Sqrt(SumSquares);
}

void FMatrix2D::Clip(float MinValue, float MaxValue)
{
    for (float& Val : Data)
    {
        Val = FMath::Clamp(Val, MinValue, MaxValue);
    }
}

void FMatrix2D::Init(float Value)
{
    for (float& Val : Data)
    {
        Val = Value;
    }
}

// Utility
FString FMatrix2D::ToString() const
{
    FString Out;
    for (int32 r = 0; r < Rows; ++r)
    {
        for (int32 c = 0; c < Columns; ++c)
        {
            float Val = Data[r * Columns + c];
            Out += FString::Printf(TEXT("%6.2f "), Val);
        }
        Out += TEXT("\n");
    }
    return Out;
}

// Dimensions
int32 FMatrix2D::GetNumRows() const
{
    return Rows;
}

int32 FMatrix2D::GetNumColumns() const
{
    return Columns;
}

// Min/Max
float FMatrix2D::Min() const
{
    check(Data.Num() > 0);

    float CurrentMin = Data[0];
    for (float Val : Data)
    {
        if (Val < CurrentMin)
        {
            CurrentMin = Val;
        }
    }
    return CurrentMin;
}

float FMatrix2D::Max() const
{
    check(Data.Num() > 0);

    float CurrentMax = Data[0];
    for (float Val : Data)
    {
        if (Val > CurrentMax)
        {
            CurrentMax = Val;
        }
    }
    return CurrentMax;
}

FMatrix2D FMatrix2D::Abs() const
{
    FMatrix2D Result(Rows, Columns);
    for (int32 i = 0; i < Data.Num(); ++i)
    {
        Result.Data[i] = FMath::Abs(Data[i]);
    }
    return Result;
}

float FMatrix2D::Sum() const
{
    float sum = 0.0f;
    for (float Val : Data)
    {
        sum += Val;
    }
    return sum;
}

float FMatrix2D::Mean() const
{
    if (Rows == 0 || Columns == 0)
    {
        return 0.0f;
    }
    return Sum() / static_cast<float>(Rows * Columns);
}

FMatrix2D FMatrix2D::T() const
{
    // Create a new matrix with swapped dimensions
    FMatrix2D Result(Columns, Rows);

    // We loop over the original (r, c). 
    // In the transposed matrix, that goes to (c, r).
    for (int32 r = 0; r < Rows; ++r)
    {
        for (int32 c = 0; c < Columns; ++c)
        {
            Result[c][r] = Data[LinearIndex(r, c)];
        }
    }
    return Result;
}

FMatrix2D FMatrix2D::MatMul(const FMatrix2D& Other) const
{
    // Standard dimension check: (A.Rows x A.Cols) * (B.Rows x B.Cols)
    check(Columns == Other.Rows);

    FMatrix2D Result(Rows, Other.Columns);

    // For each row in A
    for (int32 i = 0; i < Rows; ++i)
    {
        // For each column in B
        for (int32 j = 0; j < Other.Columns; ++j)
        {
            float Sum = 0.0f;

            // Dot product of row i of A with column j of B
            for (int32 k = 0; k < Columns; ++k)
            {
                // A(i,k) = Data[i*Columns + k]
                // B(k,j) = Other.Data[k*Other.Columns + j]
                Sum += (*this)[i][k] * Other[k][j];
            }

            Result[i][j] = Sum;
        }
    }

    return Result;
}
