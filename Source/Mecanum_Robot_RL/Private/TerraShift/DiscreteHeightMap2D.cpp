#include "TerraShift/DiscreteHeightMap2D.h"

UDiscreteHeightMap2D::UDiscreteHeightMap2D()
    : GridSizeX(0)
    , GridSizeY(0)
    , NumAgents(0)
    , MatrixDeltaRange(FVector2D::ZeroVector)
    , MaxAbsMatrixHeight(10.0f) // default
{
}

void UDiscreteHeightMap2D::Initialize(
    int32 InGridSizeX,
    int32 InGridSizeY,
    int32 InNumAgents,
    FVector2D InMatrixDeltaRange,
    float InMaxAbsMatrixHeight)
{
    GridSizeX = InGridSizeX;
    GridSizeY = InGridSizeY;
    NumAgents = InNumAgents;
    MatrixDeltaRange = InMatrixDeltaRange;
    MaxAbsMatrixHeight = InMaxAbsMatrixHeight;

    // Construct an FMatrix2D of dimension (GridSizeY x GridSizeX)
    // Initialize all values to 0.0
    HeightMap = FMatrix2D(GridSizeY, GridSizeX, 0.0f);

    // Place or randomize agent positions
    PlaceAgents();
}

void UDiscreteHeightMap2D::Reset(int32 NewNumAgents)
{
    NumAgents = NewNumAgents;

    // Re-init the heightmap to zero
    HeightMap = HeightMap.Random(-MaxAbsMatrixHeight, MaxAbsMatrixHeight);

    // Re-place agents 
    AgentPositions.SetNum(NumAgents);
    PlaceAgents();
}

void UDiscreteHeightMap2D::Update(const TArray<FAgentHeightDelta>& Deltas)
{
    // Expecting one delta per agent
    if (Deltas.Num() != NumAgents)
    {
        UE_LOG(LogTemp, Warning, TEXT("Update: #Deltas=%d != NumAgents=%d"), Deltas.Num(), NumAgents);
        return;
    }

    // For each agent, apply direction => move, then matrix update => adjust height
    for (int32 i = 0; i < NumAgents; i++)
    {
        const FAgentHeightDelta& Delta = Deltas[i];
        FIntPoint& Pos = AgentPositions[i];

        // 1) Reflect current position
        int tmp;
        switch (Delta.Reflect)
        {
        case EAgentReflection::Reflect:
            tmp = Pos.Y;
            Pos.Y = Pos.X;
            Pos.X = tmp;
            break;
        case EAgentReflection::None:
            break;
        default:
            // No movement
            break;
        }


        // 2) Increment position with wrap-around
        switch (Delta.Direction)
        {
        case EAgentDirection::Up:
            Pos.Y -= 1;
            Pos.Y = WrapY(Pos.Y);
            break;
        case EAgentDirection::Down:
            Pos.Y += 1;
            Pos.Y = WrapY(Pos.Y);
            break;
        case EAgentDirection::Left:
            Pos.X -= 1;
            Pos.X = WrapX(Pos.X);
            break;
        case EAgentDirection::Right:
            Pos.X += 1;
            Pos.X = WrapX(Pos.X);
            break;
        case EAgentDirection::None:
            break;
        default:
            // No movement
            break;
        }

        // 3) Adjust height in the cell. 
        float& CellRef = HeightMap[Pos.Y][Pos.X];
        float NewVal = CellRef;

        switch (Delta.MatrixUpdate)
        {
        case EAgentMatrixUpdate::Inc:
            NewVal += MatrixDeltaRange.Y;  // Increase by 'Y' portion
            break;
        case EAgentMatrixUpdate::Dec:
            NewVal += MatrixDeltaRange.X;  // Decrease by 'X' portion
            break;
        case EAgentMatrixUpdate::None:
            break;
        default:
            // do nothing
            break;
        }

        // 3) clamp final value to +/- MaxAbsMatrixHeight
        CellRef = FMath::Clamp(NewVal, -MaxAbsMatrixHeight, MaxAbsMatrixHeight);
    }
}

const FMatrix2D& UDiscreteHeightMap2D::GetHeights() const
{
    return HeightMap;
}

TArray<float> UDiscreteHeightMap2D::GetAgentState(int32 AgentIndex) const
{
    TArray<float> Result;
    if (AgentIndex < 0 || AgentIndex >= NumAgents)
    {
        // Return empty if invalid
        return Result;
    }

    FIntPoint Pos = AgentPositions[AgentIndex];
    // read the height from the cell
    float H = HeightMap[Pos.Y][Pos.X];

    // x, y, height
    Result.Add((float)Pos.X);
    Result.Add((float)Pos.Y);
    Result.Add(H);

    return Result;
}

void UDiscreteHeightMap2D::PlaceAgents()
{
    AgentPositions.SetNum(NumAgents);

    for (int32 i = 0; i < NumAgents; i++)
    {
        int32 RandX = FMath::RandRange(0, GridSizeX - 1);
        int32 RandY = FMath::RandRange(0, GridSizeY - 1);
        AgentPositions[i] = FIntPoint(RandX, RandY);
    }
}

int32 UDiscreteHeightMap2D::WrapX(int32 X) const
{
    // If X < 0 => wrap around to GridSizeX-1
    // If X >= GridSizeX => wrap to 0
    // A quick approach is to do modular arithmetic
    // but also handle negative in a standard way:
    //   ( (X % GridSizeX) + GridSizeX ) % GridSizeX
    const int32 modX = ((X % GridSizeX) + GridSizeX) % GridSizeX;
    return modX;
}

int32 UDiscreteHeightMap2D::WrapY(int32 Y) const
{
    const int32 modY = ((Y % GridSizeY) + GridSizeY) % GridSizeY;
    return modY;
}
