#pragma once

#include "CoreMinimal.h"
#include "Math/UnrealMathUtility.h"
#include "Matrix2D.generated.h"

// Forward declarations for proxy structs
struct FFMatrix2DRowProxy;
struct FFMatrix2DConstRowProxy;

/**
 * A simple 2D matrix struct for storing and operating on float data,
 * implemented via a single TArray<float> to maintain data locality and avoid nested TArray issues.
 * Includes checks to maintain consistency between dimensions and data storage size.
 */
USTRUCT(BlueprintType)
struct UNREALRLLABS_API FMatrix2D
{
	GENERATED_BODY()

private:
	/** Number of rows. Should only be modified internally or via Resize. */
	UPROPERTY()
	int32 Rows;

	/** Number of columns. Should only be modified internally or via Resize. */
	UPROPERTY()
	int32 Columns;

	/** Internal helper to check if the matrix state is consistent (Data.Num() == Rows * Columns). */
	FORCEINLINE void CheckInvariants(const TCHAR* Context = TEXT("")) const
	{
		checkf(Rows >= 0, TEXT("Matrix Invariant Check Failed (%s): Rows (%d) cannot be negative."), Context, Rows);
		checkf(Columns >= 0, TEXT("Matrix Invariant Check Failed (%s): Columns (%d) cannot be negative."), Context, Columns);
		const int32 ExpectedSize = Rows * Columns;
		checkf(Data.Num() == ExpectedSize, TEXT("Matrix Invariant Check Failed (%s): Data.Num() (%d) does not match Rows*Columns (%d * %d = %d)."), Context, Data.Num(), Rows, Columns, ExpectedSize);
	}

	/** Internal helper to check dimensions match for binary operations */
	FORCEINLINE void CheckDimensionsMatch(const FMatrix2D& Other, const TCHAR* Context = TEXT("")) const
	{
		checkf(Rows == Other.Rows && Columns == Other.Columns, TEXT("Matrix Dimension Mismatch (%s): (%d x %d) vs (%d x %d)"), Context, Rows, Columns, Other.Rows, Other.Columns);
	}

	/** Helper to get the linear index in the Data array for (row,col). Assumes valid indices. */
	FORCEINLINE int32 LinearIndex(int32 RowIndex, int32 ColIndex) const
	{
		// No bounds check here for performance; should be checked before calling if necessary.
		return (RowIndex * Columns) + ColIndex;
	}

public:
	/** Enum defining how Resize initializes new elements or handles existing ones. */
	enum class EInitialization { None, Zero, Uninitialized }; // Moved declaration before usage

	// Constructors & Initialization

	/** Default constructor: Creates an empty 0x0 matrix. */
	FMatrix2D();

	/** Constructor: Creates a matrix with specified dimensions, uninitialized elements. */
	FMatrix2D(int32 InRows, int32 InColumns);

	/** Constructor: Creates a matrix with specified dimensions, initialized to a specific value. */
	FMatrix2D(int32 InRows, int32 InColumns, float InitialValue);

	/** Constructor: Creates a matrix from a 2D TArray. Input must be rectangular. */
	explicit FMatrix2D(const TArray<TArray<float>>& InData);

	// Copy/Move Semantics

	FMatrix2D(const FMatrix2D& Other);
	FMatrix2D& operator=(const FMatrix2D& Other);
	FMatrix2D(FMatrix2D&& Other) noexcept;
	FMatrix2D& operator=(FMatrix2D&& Other) noexcept;

	// Element-wise Operators (Matrix-Matrix)

	[[nodiscard]] FMatrix2D operator+(const FMatrix2D& Other) const;
	FMatrix2D& operator+=(const FMatrix2D& Other);
	[[nodiscard]] FMatrix2D operator-(const FMatrix2D& Other) const;
	FMatrix2D& operator-=(const FMatrix2D& Other);
	/** Element-wise multiplication (Hadamard product). */
	[[nodiscard]] FMatrix2D operator*(const FMatrix2D& Other) const;
	FMatrix2D& operator*=(const FMatrix2D& Other);
	/** Element-wise division. Checks for division by zero. */
	[[nodiscard]] FMatrix2D operator/(const FMatrix2D& Other) const;
	FMatrix2D& operator/=(const FMatrix2D& Other);

	// Scalar Operators

	[[nodiscard]] FMatrix2D operator+(float Scalar) const;
	FMatrix2D& operator+=(float Scalar);
	[[nodiscard]] FMatrix2D operator-(float Scalar) const;
	FMatrix2D& operator-=(float Scalar);
	[[nodiscard]] FMatrix2D operator*(float Scalar) const;
	FMatrix2D& operator*=(float Scalar);
	/** Scalar division. Checks for division by zero. */
	[[nodiscard]] FMatrix2D operator/(float Scalar) const;
	FMatrix2D& operator/=(float Scalar);

	// Friend operators for scalar on the left
	friend FMatrix2D operator+(float Scalar, const FMatrix2D& Matrix);
	friend FMatrix2D operator-(float Scalar, const FMatrix2D& Matrix);
	friend FMatrix2D operator*(float Scalar, const FMatrix2D& Matrix);
	friend FMatrix2D operator/(float Scalar, const FMatrix2D& Matrix);

	// Accessors & Indexing

	/** Non-const row indexing: returns a proxy object for M[row][col] access. */
	FFMatrix2DRowProxy operator[](int32 RowIndex);
	/** Const row indexing: returns a const proxy object for M[row][col] access. */
	const FFMatrix2DConstRowProxy operator[](int32 RowIndex) const;

	/** Get the number of rows. */
	FORCEINLINE int32 GetNumRows() const { return Rows; }
	/** Get the number of columns. */
	FORCEINLINE int32 GetNumColumns() const { return Columns; }
	/** Get the total number of elements (Rows * Columns). */
	FORCEINLINE int32 Num() const { return Rows * Columns; }

	/** Get read-only access to the underlying flat data array. */
	const TArray<float>& GetData() const { return Data; }
	/** Get read/write access to the underlying flat data array. Use with extreme caution, as direct modification can break invariants. Prefer using matrix methods. */
	TArray<float>& GetData_Unsafe() { return Data; }

	// Matrix Operations & Manipulation

	/** Extracts a sub-matrix. Indices are inclusive. Negative indices count from the end. */
	[[nodiscard]] FMatrix2D Sub(int32 RowStart, int32 RowEnd, int32 ColStart, int32 ColEnd) const;

	/** Applies the exponential function element-wise. */
	[[nodiscard]] FMatrix2D Exp() const;
	/** Applies the cosine function element-wise. */
	[[nodiscard]] FMatrix2D Cos() const;
	/** Applies the sine function element-wise. */
	[[nodiscard]] FMatrix2D Sin() const;
	/** Applies the hyperbolic tangent function element-wise. */
	[[nodiscard]] FMatrix2D Tanh() const;
	/** Applies the absolute value function element-wise. */
	[[nodiscard]] FMatrix2D Abs() const;

	/** Calculates the element-wise dot product (sum of Hadamard product) with another matrix of the same dimensions. */
	[[nodiscard]] float Dot(const FMatrix2D& Other) const;
	/** Calculates the Frobenius norm (sqrt of sum of squares). */
	[[nodiscard]] float Norm() const;
	/** Calculates the sum of all elements. */
	[[nodiscard]] float Sum() const;
	/** Calculates the mean of all elements. Returns 0 for an empty matrix. */
	[[nodiscard]] float Mean() const;
	/** Finds the minimum element value. Asserts if the matrix is empty. */
	[[nodiscard]] float Min() const;
	/** Finds the maximum element value. Asserts if the matrix is empty. */
	[[nodiscard]] float Max() const;

	/** Clamps all elements within the specified range [MinValue, MaxValue]. */
	void Clip(float MinValue, float MaxValue);
	/** Initializes all elements to the specified value. */
	void Init(float Value);

	/** Returns a string representation of the matrix (potentially truncated for large matrices). */
	[[nodiscard]] FString ToString() const;
	/** Creates a new matrix with the same dimensions, filled with random numbers in the range [Min, Max]. */
	[[nodiscard]] FMatrix2D Random(float Min, float Max) const;

	/** Resizes the matrix to NewRows x NewCols. Data may be lost or uninitialized depending on InitMethod. */
	void Resize(int32 NewRows, int32 NewCols, EInitialization InitMethod = EInitialization::None); // Enum declared above

	/** Returns the transpose of the matrix. */
	[[nodiscard]] FMatrix2D T() const;

	/** Performs standard matrix multiplication (this * Other). Asserts if dimensions are incompatible. */
	[[nodiscard]] FMatrix2D MatMul(const FMatrix2D& Other) const;

public: // Keep Data public for UPROPERTY reflection, but discourage direct modification.
	/**
	 * Single TArray holding the matrix data in row-major order.
	 * Size should always be Rows * Columns. Direct modification can break class invariants.
	 * Prefer using matrix methods like Resize, Init, Clip, or operators.
	 */
	UPROPERTY()
	TArray<float> Data;
};


//============================================================================
// Proxy Struct Definitions (moved implementation outside FMatrix2D for clarity)
//============================================================================

/**
 * A "row proxy" that allows user-friendly non-const [row][col] access while
 * internally referencing a single TArray<float> storage.
 */
struct FFMatrix2DRowProxy
{
	friend struct FMatrix2D;
	friend struct FFMatrix2DConstRowProxy;

private:
	float* RowData; // Pointer to the start of the row in the parent matrix's Data array
	int32 NumCols;  // Number of columns in the parent matrix

	// Private constructor: Only FMatrix2D can create this proxy.
	FFMatrix2DRowProxy(float* InRowData, int32 InNumCols)
		: RowData(InRowData), NumCols(InNumCols) {}

public:
	// Column indexing operator: Returns a modifiable reference to the element.
	float& operator[](int32 ColIndex)
	{
		checkf(ColIndex >= 0 && ColIndex < NumCols, TEXT("Column index %d out of bounds (0-%d)"), ColIndex, NumCols - 1);
		checkf(RowData != nullptr, TEXT("RowData pointer is null in FFMatrix2DRowProxy"));
		return RowData[ColIndex];
	}

	// Column indexing operator (const overload): Returns a const reference (allows reading from non-const proxy).
	const float& operator[](int32 ColIndex) const
	{
		checkf(ColIndex >= 0 && ColIndex < NumCols, TEXT("Column index %d out of bounds (0-%d)"), ColIndex, NumCols - 1);
		checkf(RowData != nullptr, TEXT("RowData pointer is null in FFMatrix2DRowProxy"));
		return RowData[ColIndex];
	}
};

/**
 * A "const row proxy" that allows user-friendly const [row][col] access.
 */
struct FFMatrix2DConstRowProxy
{
	friend struct FMatrix2D;
private:
	const float* RowData; // Const pointer to the start of the row
	int32 NumCols;       // Number of columns

	// Private constructor: Only FMatrix2D can create this proxy.
	FFMatrix2DConstRowProxy(const float* InRowData, int32 InNumCols)
		: RowData(InRowData), NumCols(InNumCols) {}

public:
	// Column indexing operator: Returns a const reference to the element.
	const float& operator[](int32 ColIndex) const
	{
		checkf(ColIndex >= 0 && ColIndex < NumCols, TEXT("Column index %d out of bounds (0-%d)"), ColIndex, NumCols - 1);
		checkf(RowData != nullptr, TEXT("RowData pointer is null in FFMatrix2DConstRowProxy"));
		return RowData[ColIndex];
	}
};
