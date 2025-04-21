#include "TerraShift/Matrix2D.h"
#include "Templates/UnrealTemplate.h" // For MoveTemp

//============================================================================
// Constructors & Destructor
//============================================================================

FMatrix2D::FMatrix2D()
	: Rows(0)
	, Columns(0)
{
	// Default constructor creates an empty 0x0 matrix.
	CheckInvariants(TEXT("Default Constructor"));
}

FMatrix2D::FMatrix2D(int32 InRows, int32 InColumns)
	: Rows(InRows)
	, Columns(InColumns)
{
	checkf(InRows >= 0 && InColumns >= 0, TEXT("Matrix dimensions cannot be negative (%d, %d)"), InRows, InColumns);
	// Allocate memory without initializing elements for performance.
	Data.SetNumUninitialized(Rows * Columns);
	CheckInvariants(TEXT("Dims Constructor"));
}

FMatrix2D::FMatrix2D(int32 InRows, int32 InColumns, float InitialValue)
	: Rows(InRows)
	, Columns(InColumns)
{
	checkf(InRows >= 0 && InColumns >= 0, TEXT("Matrix dimensions cannot be negative (%d, %d)"), InRows, InColumns);
	// Allocate memory and initialize all elements to InitialValue.
	Data.Init(InitialValue, Rows * Columns);
	CheckInvariants(TEXT("Dims+Value Constructor"));
}

FMatrix2D::FMatrix2D(const TArray<TArray<float>>& InData)
{
	Rows = InData.Num();
	Columns = (Rows > 0) ? InData[0].Num() : 0;
	checkf(Rows >= 0 && Columns >= 0, TEXT("Matrix dimensions cannot be negative (%d, %d) from 2D Array"), Rows, Columns);

	const int32 ExpectedSize = Rows * Columns;
	Data.SetNumUninitialized(ExpectedSize); // Allocate space

	for (int32 r = 0; r < Rows; ++r)
	{
		// Ensure the input TArray is rectangular.
		checkf(InData[r].Num() == Columns, TEXT("Input 2D Array is not rectangular at row %d. Expected %d columns, got %d."), r, Columns, InData[r].Num());
		// Efficiently copy data row by row.
		FMemory::Memcpy(&Data[LinearIndex(r, 0)], InData[r].GetData(), Columns * sizeof(float));
	}
	CheckInvariants(TEXT("2D Array Constructor"));
}

//============================================================================
// Copy/Move Semantics
//============================================================================

FMatrix2D::FMatrix2D(const FMatrix2D& Other)
	: Rows(Other.Rows)
	, Columns(Other.Columns)
	, Data(Other.Data) // TArray handles deep copy.
{
	// Assume 'Other' is valid; check the newly constructed matrix.
	CheckInvariants(TEXT("Copy Constructor"));
}

FMatrix2D& FMatrix2D::operator=(const FMatrix2D& Other)
{
	if (this != &Other)
	{
		Other.CheckInvariants(TEXT("Assignment Operator (Source)"));

		Rows = Other.Rows;
		Columns = Other.Columns;
		Data = Other.Data; // TArray handles deep copy assignment.

		CheckInvariants(TEXT("Assignment Operator (Dest)"));
	}
	return *this;
}


FMatrix2D::FMatrix2D(FMatrix2D&& Other) noexcept
// Use direct assignment and reset for POD types instead of Exchange with literal 0
	: Rows(Other.Rows)
	, Columns(Other.Columns)
	, Data(MoveTemp(Other.Data))          // Move TArray data ownership.
{
	// Reset source POD members after taking ownership
	Other.Rows = 0;
	Other.Columns = 0;

	CheckInvariants(TEXT("Move Constructor (Dest)"));
	// Ensure the source matrix is left in a valid empty state.
	Other.CheckInvariants(TEXT("Move Constructor (Source After Move)"));
}

FMatrix2D& FMatrix2D::operator=(FMatrix2D&& Other) noexcept
{
	if (this != &Other)
	{
		Other.CheckInvariants(TEXT("Move Assignment (Source Before Move)"));

		// TArray's move assignment will handle releasing existing resources.
		Data = MoveTemp(Other.Data);
		// Use direct assignment and reset for POD types
		Rows = Other.Rows;
		Columns = Other.Columns;
		Other.Rows = 0;
		Other.Columns = 0;


		CheckInvariants(TEXT("Move Assignment (Dest)"));
		// Ensure the source matrix is left in a valid empty state.
		Other.CheckInvariants(TEXT("Move Assignment (Source After Move)"));
	}
	return *this;
}

//============================================================================
// Core Operations
//============================================================================

// Use FMatrix2D::EInitialization for the enum type
void FMatrix2D::Resize(int32 NewRows, int32 NewCols, FMatrix2D::EInitialization InitMethod)
{
	checkf(NewRows >= 0 && NewCols >= 0, TEXT("Cannot resize matrix to negative dimensions (%d, %d)"), NewRows, NewCols);

	// Calculate required size *before* modifying member variables
	const int32 NewSize = NewRows * NewCols;

	// Now update member variables
	Rows = NewRows;
	Columns = NewCols;

	switch (InitMethod)
	{
		// Use FMatrix2D::EInitialization:: for enum values
	case FMatrix2D::EInitialization::Zero:
		Data.Init(0.0f, NewSize); // Initialize new elements to zero.
		break;
	case FMatrix2D::EInitialization::Uninitialized:
		// Allocate memory without initializing new elements. Faster but potentially unsafe if not handled carefully.
		// Note: SetNumUninitialized doesn't shrink memory allocation if NewSize < CurrentMax. Use SetNum if shrinking is needed.
		Data.SetNumUninitialized(NewSize);
		break;
	case FMatrix2D::EInitialization::None:
	default:
		// Standard TArray::SetNum behavior (typically zeros new elements, preserves existing).
		Data.SetNum(NewSize);
		break;
	}

	// Ensure state is consistent after resize.
	CheckInvariants(TEXT("Resize"));
}


//============================================================================
// Element-wise Operators (Matrix-Matrix)
//============================================================================

FMatrix2D FMatrix2D::operator+(const FMatrix2D& Other) const
{
	CheckInvariants(TEXT("Operator + (this)"));
	Other.CheckInvariants(TEXT("Operator + (Other)"));
	CheckDimensionsMatch(Other, TEXT("Operator +"));

	FMatrix2D Result(Rows, Columns); // Constructor handles allocation and invariant check.
	const int32 ExpectedSize = Num();

	for (int32 i = 0; i < ExpectedSize; ++i)
	{
		Result.Data[i] = Data[i] + Other.Data[i];
	}
	return Result;
}

FMatrix2D& FMatrix2D::operator+=(const FMatrix2D& Other)
{
	CheckInvariants(TEXT("Operator += (this)"));
	Other.CheckInvariants(TEXT("Operator += (Other)"));
	CheckDimensionsMatch(Other, TEXT("Operator +="));

	const int32 ExpectedSize = Num();
	for (int32 i = 0; i < ExpectedSize; ++i)
	{
		Data[i] += Other.Data[i];
	}
	return *this;
}

FMatrix2D FMatrix2D::operator-(const FMatrix2D& Other) const
{
	CheckInvariants(TEXT("Operator - (this)"));
	Other.CheckInvariants(TEXT("Operator - (Other)"));
	CheckDimensionsMatch(Other, TEXT("Operator -"));

	FMatrix2D Result(Rows, Columns);
	const int32 ExpectedSize = Num();
	for (int32 i = 0; i < ExpectedSize; ++i)
	{
		Result.Data[i] = Data[i] - Other.Data[i];
	}
	return Result;
}

FMatrix2D& FMatrix2D::operator-=(const FMatrix2D& Other)
{
	CheckInvariants(TEXT("Operator -= (this)"));
	Other.CheckInvariants(TEXT("Operator -= (Other)"));
	CheckDimensionsMatch(Other, TEXT("Operator -="));

	const int32 ExpectedSize = Num();
	for (int32 i = 0; i < ExpectedSize; ++i)
	{
		Data[i] -= Other.Data[i];
	}
	return *this;
}

// Element-wise multiplication (Hadamard product)
FMatrix2D FMatrix2D::operator*(const FMatrix2D& Other) const
{
	CheckInvariants(TEXT("Operator * (Element-wise) (this)"));
	Other.CheckInvariants(TEXT("Operator * (Element-wise) (Other)"));
	CheckDimensionsMatch(Other, TEXT("Operator * (Element-wise)"));

	FMatrix2D Result(Rows, Columns);
	const int32 ExpectedSize = Num();
	for (int32 i = 0; i < ExpectedSize; ++i)
	{
		Result.Data[i] = Data[i] * Other.Data[i];
	}
	return Result;
}

FMatrix2D& FMatrix2D::operator*=(const FMatrix2D& Other)
{
	CheckInvariants(TEXT("Operator *= (Element-wise) (this)"));
	Other.CheckInvariants(TEXT("Operator *= (Element-wise) (Other)"));
	CheckDimensionsMatch(Other, TEXT("Operator *= (Element-wise)"));

	const int32 ExpectedSize = Num();
	for (int32 i = 0; i < ExpectedSize; ++i)
	{
		Data[i] *= Other.Data[i];
	}
	return *this;
}

// Element-wise division
FMatrix2D FMatrix2D::operator/(const FMatrix2D& Other) const
{
	CheckInvariants(TEXT("Operator / (Element-wise) (this)"));
	Other.CheckInvariants(TEXT("Operator / (Element-wise) (Other)"));
	CheckDimensionsMatch(Other, TEXT("Operator / (Element-wise)"));

	FMatrix2D Result(Rows, Columns);
	const int32 ExpectedSize = Num();
	for (int32 i = 0; i < ExpectedSize; ++i)
	{
		checkf(!FMath::IsNearlyZero(Other.Data[i]), TEXT("Element-wise division by zero at index %d"), i);
		Result.Data[i] = Data[i] / Other.Data[i];
	}
	return Result;
}

FMatrix2D& FMatrix2D::operator/=(const FMatrix2D& Other)
{
	CheckInvariants(TEXT("Operator /= (Element-wise) (this)"));
	Other.CheckInvariants(TEXT("Operator /= (Element-wise) (Other)"));
	CheckDimensionsMatch(Other, TEXT("Operator /= (Element-wise)"));

	const int32 ExpectedSize = Num();
	for (int32 i = 0; i < ExpectedSize; ++i)
	{
		checkf(!FMath::IsNearlyZero(Other.Data[i]), TEXT("Element-wise division by zero at index %d"), i);
		Data[i] /= Other.Data[i];
	}
	return *this;
}

//============================================================================
// Scalar Operators
//============================================================================

FMatrix2D FMatrix2D::operator+(float Scalar) const
{
	CheckInvariants(TEXT("Operator + (Scalar)"));
	FMatrix2D Result(Rows, Columns);
	const int32 ExpectedSize = Num();
	for (int32 i = 0; i < ExpectedSize; ++i)
	{
		Result.Data[i] = Data[i] + Scalar;
	}
	return Result;
}

FMatrix2D& FMatrix2D::operator+=(float Scalar)
{
	CheckInvariants(TEXT("Operator += (Scalar)"));
	const int32 ExpectedSize = Num();
	for (int32 i = 0; i < ExpectedSize; ++i)
	{
		Data[i] += Scalar;
	}
	return *this;
}

FMatrix2D FMatrix2D::operator-(float Scalar) const
{
	CheckInvariants(TEXT("Operator - (Scalar)"));
	FMatrix2D Result(Rows, Columns);
	const int32 ExpectedSize = Num();
	for (int32 i = 0; i < ExpectedSize; ++i)
	{
		Result.Data[i] = Data[i] - Scalar;
	}
	return Result;
}

FMatrix2D& FMatrix2D::operator-=(float Scalar)
{
	CheckInvariants(TEXT("Operator -= (Scalar)"));
	const int32 ExpectedSize = Num();
	for (int32 i = 0; i < ExpectedSize; ++i)
	{
		Data[i] -= Scalar;
	}
	return *this;
}

FMatrix2D FMatrix2D::operator*(float Scalar) const
{
	CheckInvariants(TEXT("Operator * (Scalar)"));
	FMatrix2D Result(Rows, Columns);
	const int32 ExpectedSize = Num();
	for (int32 i = 0; i < ExpectedSize; ++i)
	{
		Result.Data[i] = Data[i] * Scalar;
	}
	return Result;
}

FMatrix2D& FMatrix2D::operator*=(float Scalar)
{
	CheckInvariants(TEXT("Operator *= (Scalar)"));
	const int32 ExpectedSize = Num();
	for (int32 i = 0; i < ExpectedSize; ++i)
	{
		Data[i] *= Scalar;
	}
	return *this;
}

FMatrix2D FMatrix2D::operator/(float Scalar) const
{
	CheckInvariants(TEXT("Operator / (Scalar)"));
	checkf(!FMath::IsNearlyZero(Scalar), TEXT("Division by zero scalar"));
	FMatrix2D Result(Rows, Columns);
	const float InvScalar = 1.0f / Scalar; // Multiply by inverse for potential performance gain
	const int32 ExpectedSize = Num();
	for (int32 i = 0; i < ExpectedSize; ++i)
	{
		Result.Data[i] = Data[i] * InvScalar;
	}
	return Result;
}

FMatrix2D& FMatrix2D::operator/=(float Scalar)
{
	CheckInvariants(TEXT("Operator /= (Scalar)"));
	checkf(!FMath::IsNearlyZero(Scalar), TEXT("Division by zero scalar"));
	const float InvScalar = 1.0f / Scalar; // Multiply by inverse
	const int32 ExpectedSize = Num();
	for (int32 i = 0; i < ExpectedSize; ++i)
	{
		Data[i] *= InvScalar;
	}
	return *this;
}

//============================================================================
// Friend Scalar Operators (Scalar on Left)
//============================================================================

FMatrix2D operator+(float Scalar, const FMatrix2D& Matrix)
{
	// Reuse existing operator+
	return Matrix + Scalar;
}

FMatrix2D operator-(float Scalar, const FMatrix2D& Matrix)
{
	Matrix.CheckInvariants(TEXT("Operator - (Scalar Left)"));
	FMatrix2D Result(Matrix.Rows, Matrix.Columns);
	const int32 ExpectedSize = Matrix.Num();
	for (int32 i = 0; i < ExpectedSize; ++i)
	{
		Result.Data[i] = Scalar - Matrix.Data[i];
	}
	return Result;
}

FMatrix2D operator*(float Scalar, const FMatrix2D& Matrix)
{
	// Reuse existing operator*
	return Matrix * Scalar;
}

FMatrix2D operator/(float Scalar, const FMatrix2D& Matrix)
{
	Matrix.CheckInvariants(TEXT("Operator / (Scalar Left)"));
	FMatrix2D Result(Matrix.Rows, Matrix.Columns);
	const int32 ExpectedSize = Matrix.Num();
	for (int32 i = 0; i < ExpectedSize; ++i)
	{
		checkf(!FMath::IsNearlyZero(Matrix.Data[i]), TEXT("Scalar division by zero matrix element at index %d"), i);
		Result.Data[i] = Scalar / Matrix.Data[i];
	}
	return Result;
}

//============================================================================
// Indexing Operators
//============================================================================

FFMatrix2DRowProxy FMatrix2D::operator[](int32 RowIndex)
{
	// Optional: Check invariants before allowing access. Can be noisy.
	// CheckInvariants(TEXT("Operator [] (Non-const)"));

	// Support negative indexing (e.g., -1 for last row)
	if (RowIndex < 0)
	{
		RowIndex += Rows;
	}
	checkf(RowIndex >= 0 && RowIndex < Rows, TEXT("Row index %d out of bounds (0-%d)"), RowIndex, Rows - 1);

	// Return the proxy object pointing to the start of the requested row.
	return FFMatrix2DRowProxy(&Data[LinearIndex(RowIndex, 0)], Columns);
}

const FFMatrix2DConstRowProxy FMatrix2D::operator[](int32 RowIndex) const
{
	// Optional: Check invariants before allowing access. Can be noisy.
	// CheckInvariants(TEXT("Operator [] (Const)"));

	if (RowIndex < 0)
	{
		RowIndex += Rows;
	}
	checkf(RowIndex >= 0 && RowIndex < Rows, TEXT("Row index %d out of bounds (0-%d)"), RowIndex, Rows - 1);

	// Return the const proxy object pointing to the start of the requested row.
	return FFMatrix2DConstRowProxy(&Data[LinearIndex(RowIndex, 0)], Columns);
}

//============================================================================
// Sub-Matrix Extraction
//============================================================================

FMatrix2D FMatrix2D::Sub(int32 RowStart, int32 RowEnd, int32 ColStart, int32 ColEnd) const
{
	CheckInvariants(TEXT("Sub Matrix (Source)"));

	// Adjust negative indices to be relative to the end.
	if (RowStart < 0) RowStart += Rows;
	if (RowEnd < 0)   RowEnd += Rows;
	if (ColStart < 0) ColStart += Columns;
	if (ColEnd < 0)   ColEnd += Columns;

	// Validate the final indices after potential adjustment.
	checkf(RowStart >= 0 && RowStart < Rows, TEXT("Sub matrix RowStart %d out of bounds (0-%d)"), RowStart, Rows - 1);
	checkf(RowEnd >= RowStart && RowEnd < Rows, TEXT("Sub matrix RowEnd %d invalid (must be >= RowStart %d and < Rows %d)"), RowEnd, RowStart, Rows);
	checkf(ColStart >= 0 && ColStart < Columns, TEXT("Sub matrix ColStart %d out of bounds (0-%d)"), ColStart, Columns - 1);
	checkf(ColEnd >= ColStart && ColEnd < Columns, TEXT("Sub matrix ColEnd %d invalid (must be >= ColStart %d and < Columns %d)"), ColEnd, ColStart, Columns);

	const int32 NewRows = RowEnd - RowStart + 1;
	const int32 NewCols = ColEnd - ColStart + 1;

	FMatrix2D Result(NewRows, NewCols); // Constructor checks invariants.
	for (int32 r = 0; r < NewRows; ++r)
	{
		// Use Memcpy for efficiency when copying contiguous row segments.
		const int32 SrcLinearIndex = LinearIndex(RowStart + r, ColStart);
		const int32 DestLinearIndex = Result.LinearIndex(r, 0);
		FMemory::Memcpy(&Result.Data[DestLinearIndex], &Data[SrcLinearIndex], NewCols * sizeof(float));
	}
	return Result;
}

//============================================================================
// Mathematical Functions (Element-wise)
//============================================================================

FMatrix2D FMatrix2D::Random(float Min, float Max) const
{
	// Creates a new matrix with the same dimensions as 'this', filled with random values.
	FMatrix2D Result(Rows, Columns);
	const int32 ExpectedSize = Num();
	for (int32 i = 0; i < ExpectedSize; ++i)
	{
		Result.Data[i] = FMath::FRandRange(Min, Max);
	}
	return Result;
}

FMatrix2D FMatrix2D::Exp() const
{
	CheckInvariants(TEXT("Exp"));
	FMatrix2D Result(Rows, Columns);
	const int32 ExpectedSize = Num();
	for (int32 i = 0; i < ExpectedSize; ++i)
	{
		Result.Data[i] = FMath::Exp(Data[i]);
	}
	return Result;
}

FMatrix2D FMatrix2D::Cos() const
{
	CheckInvariants(TEXT("Cos"));
	FMatrix2D Result(Rows, Columns);
	const int32 ExpectedSize = Num();
	for (int32 i = 0; i < ExpectedSize; ++i)
	{
		Result.Data[i] = FMath::Cos(Data[i]);
	}
	return Result;
}

FMatrix2D FMatrix2D::Sin() const
{
	CheckInvariants(TEXT("Sin"));
	FMatrix2D Result(Rows, Columns);
	const int32 ExpectedSize = Num();
	for (int32 i = 0; i < ExpectedSize; ++i)
	{
		Result.Data[i] = FMath::Sin(Data[i]);
	}
	return Result;
}

FMatrix2D FMatrix2D::Tanh() const
{
	CheckInvariants(TEXT("Tanh"));
	FMatrix2D Result(Rows, Columns);
	const int32 ExpectedSize = Num();
	for (int32 i = 0; i < ExpectedSize; ++i)
	{
		Result.Data[i] = FMath::Tanh(Data[i]);
	}
	return Result;
}

FMatrix2D FMatrix2D::Abs() const
{
	CheckInvariants(TEXT("Abs"));
	FMatrix2D Result(Rows, Columns);
	const int32 ExpectedSize = Num();
	for (int32 i = 0; i < ExpectedSize; ++i)
	{
		Result.Data[i] = FMath::Abs(Data[i]);
	}
	return Result;
}

//============================================================================
// Matrix Reduction Functions
//============================================================================

// Calculates the element-wise dot product (sum of the Hadamard product).
float FMatrix2D::Dot(const FMatrix2D& Other) const
{
	CheckInvariants(TEXT("Dot (this)"));
	Other.CheckInvariants(TEXT("Dot (Other)"));
	CheckDimensionsMatch(Other, TEXT("Dot"));

	float Sum = 0.0f;
	const int32 ExpectedSize = Num();
	for (int32 i = 0; i < ExpectedSize; ++i)
	{
		Sum += Data[i] * Other.Data[i];
	}
	return Sum;
}

// Calculates the Frobenius norm (sqrt of sum of squares of elements).
float FMatrix2D::Norm() const
{
	CheckInvariants(TEXT("Norm"));
	float SumSquares = 0.0f;
	const int32 ExpectedSize = Num();
	for (int32 i = 0; i < ExpectedSize; ++i)
	{
		SumSquares += Data[i] * Data[i];
	}
	return FMath::Sqrt(SumSquares);
}

void FMatrix2D::Clip(float MinValue, float MaxValue)
{
	CheckInvariants(TEXT("Clip"));
	const int32 ExpectedSize = Num();
	for (int32 i = 0; i < ExpectedSize; ++i)
	{
		Data[i] = FMath::Clamp(Data[i], MinValue, MaxValue);
	}
}

// Initializes all elements of the matrix to a specific value.
void FMatrix2D::Init(float Value)
{
	CheckInvariants(TEXT("Init"));
	const int32 ExpectedSize = Num();
	// Use TArray's efficient Init method.
	Data.Init(Value, ExpectedSize);
	// Re-check invariants after bulk modification.
	CheckInvariants(TEXT("Init (End)"));
}

float FMatrix2D::Sum() const
{
	CheckInvariants(TEXT("Sum"));
	float sum = 0.0f;
	const int32 ExpectedSize = Num();
	for (int32 i = 0; i < ExpectedSize; ++i)
	{
		sum += Data[i];
	}
	return sum;
}

float FMatrix2D::Mean() const
{
	CheckInvariants(TEXT("Mean"));
	const int32 ExpectedSize = Num();
	if (ExpectedSize == 0)
	{
		// Define behavior for empty matrix (0 or NaN).
		return 0.0f;
	}
	return Sum() / static_cast<float>(ExpectedSize);
}

float FMatrix2D::Min() const
{
	CheckInvariants(TEXT("Min"));
	const int32 ExpectedSize = Num();
	checkf(ExpectedSize > 0, TEXT("Cannot get Min of an empty matrix"));

	// Initialize with the first element.
	float CurrentMin = Data[0];
	// Iterate starting from the second element.
	for (int32 i = 1; i < ExpectedSize; ++i)
	{
		CurrentMin = FMath::Min(CurrentMin, Data[i]);
	}
	return CurrentMin;
}

float FMatrix2D::Max() const
{
	CheckInvariants(TEXT("Max"));
	const int32 ExpectedSize = Num();
	checkf(ExpectedSize > 0, TEXT("Cannot get Max of an empty matrix"));

	// Initialize with the first element.
	float CurrentMax = Data[0];
	// Iterate starting from the second element.
	for (int32 i = 1; i < ExpectedSize; ++i)
	{
		CurrentMax = FMath::Max(CurrentMax, Data[i]);
	}
	return CurrentMax;
}

//============================================================================
// Utility Functions
//============================================================================

FString FMatrix2D::ToString() const
{
	CheckInvariants(TEXT("ToString"));
	// Provide a more informative string representation, limiting output size.
	FString Out = FString::Printf(TEXT("FMatrix2D (%d x %d):\n"), Rows, Columns);
	constexpr int MaxRowsToShow = 10;
	constexpr int MaxColsToShow = 10;

	for (int32 r = 0; r < FMath::Min(Rows, MaxRowsToShow); ++r)
	{
		Out += TEXT("[");
		for (int32 c = 0; c < FMath::Min(Columns, MaxColsToShow); ++c)
		{
			Out += FString::Printf(TEXT("% 8.3f"), Data[LinearIndex(r, c)]);
			if (c < FMath::Min(Columns, MaxColsToShow) - 1) Out += TEXT(", ");
		}
		if (Columns > MaxColsToShow) Out += TEXT(", ..."); // Indicate truncated columns
		Out += TEXT("]\n");
	}
	if (Rows > MaxRowsToShow) Out += TEXT("...\n"); // Indicate truncated rows
	return Out;
}

//============================================================================
// Matrix Operations
//============================================================================

FMatrix2D FMatrix2D::T() const
{
	CheckInvariants(TEXT("Transpose (Source)"));
	FMatrix2D Result(Columns, Rows); // Result has swapped dimensions.

	for (int32 r = 0; r < Rows; ++r)
	{
		for (int32 c = 0; c < Columns; ++c)
		{
			// Element at (r, c) in source goes to (c, r) in destination.
			Result.Data[Result.LinearIndex(c, r)] = Data[LinearIndex(r, c)];
		}
	}
	return Result;
}

FMatrix2D FMatrix2D::MatMul(const FMatrix2D& Other) const
{
	CheckInvariants(TEXT("MatMul (this)"));
	Other.CheckInvariants(TEXT("MatMul (Other)"));

	// Matrix multiplication dimension check: (A.Rows x A.Cols) * (B.Rows x B.Cols) requires A.Cols == B.Rows.
	checkf(Columns == Other.Rows, TEXT("Matrix multiplication dimension mismatch: (%d x %d) * (%d x %d). Inner dimensions must match."), Rows, Columns, Other.Rows, Other.Columns);

	FMatrix2D Result(Rows, Other.Columns); // Result dimensions are outer dimensions.

	// Standard naive matrix multiplication. Consider optimized libraries (e.g., BLAS via plugins) for performance-critical scenarios.
	for (int32 i = 0; i < Result.Rows; ++i) // Iterate rows of Result (and this)
	{
		for (int32 j = 0; j < Result.Columns; ++j) // Iterate columns of Result (and Other)
		{
			float Sum = 0.0f;
			for (int32 k = 0; k < Columns; ++k) // Iterate inner dimension (Columns of this, Rows of Other)
			{
				// Access elements directly using LinearIndex for potentially better performance than operator[].
				Sum += Data[LinearIndex(i, k)] * Other.Data[Other.LinearIndex(k, j)];
			}
			Result.Data[Result.LinearIndex(i, j)] = Sum;
		}
	}
	return Result;
}
