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
    HeightMap = FMatrix2D(GridSizeY, GridSizeX, 0.0f);

    // Place or randomize agent positions
    PlaceAgents();
}

void UDiscreteHeightMap2D::Reset(int32 NewNumAgents)
{
    NumAgents = NewNumAgents;

    // Re-init the heightmap to zero
    HeightMap.Init(0.0f);

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
        // Move agent
        FIntPoint& Pos = AgentPositions[i];

        switch (Delta.Direction)
        {
        case EAgentDirection::Up:
            Pos.Y = FMath::Clamp(Pos.Y - 1, 0, GridSizeY - 1);
            break;
        case EAgentDirection::Down:
            Pos.Y = FMath::Clamp(Pos.Y + 1, 0, GridSizeY - 1);
            break;
        case EAgentDirection::Left:
            Pos.X = FMath::Clamp(Pos.X - 1, 0, GridSizeX - 1);
            break;
        case EAgentDirection::Right:
            Pos.X = FMath::Clamp(Pos.X + 1, 0, GridSizeX - 1);
            break;
        default:
            break;
        }

        // Adjust height. 
        // We'll interpret: Rows => Y, Cols => X
        float& CellRef = HeightMap[Pos.Y][Pos.X];
        float NewVal = CellRef;

        switch (Delta.MatrixUpdate)
        {
        case EAgentMatrixUpdate::Inc:
            NewVal += MatrixDeltaRange.Y;
            break;
        case EAgentMatrixUpdate::Dec:
            NewVal += MatrixDeltaRange.X;
            break;
        case EAgentMatrixUpdate::Zero:
            NewVal = 0.0f;
            break;
        case EAgentMatrixUpdate::None:
        default:
            // do nothing
            break;
        }

        // clamp
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
        return Result;
    }

    FIntPoint Pos = AgentPositions[AgentIndex];
    const float H = HeightMap[Pos.Y][Pos.X];
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
