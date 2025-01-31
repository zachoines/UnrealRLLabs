//#include "TerraShift/DiscreteFourier2D.h"
//
//UDiscreteFourier2D::UDiscreteFourier2D()
//{
//    GridSizeX = 0;
//    GridSizeY = 0;
//    K = 0;
//}
//
//void UDiscreteFourier2D::Initialize(
//    int32 InGridSizeX,
//    int32 InGridSizeY,
//    int32 InNumAgents,
//    int32 InK,
//    FVector2D InMatrixDeltaRange,
//    float InMaxAbsMatrixMovement
//)
//{
//    GridSizeX = InGridSizeX;
//    GridSizeY = InGridSizeY;
//    K = InK;
//    MatrixDeltaRange = InMatrixDeltaRange;
//    MaxAbsMatrixMovement = InMaxAbsMatrixMovement;
//
//    // Build the standard basis Sx, Sy once
//    BuildBasis();
//
//    // Allocate agent array
//    Agents.SetNum(InNumAgents);
//
//    int32 Dim = 2 * K;
//    // For each agent, zero out AgentA and pick a random location
//    for (int32 i = 0; i < InNumAgents; ++i)
//    {
//        FAgentFourierState& Agt = Agents[i];
//        Agt.AgentA = FMatrix2D(Dim, Dim, 0.0f);
//
//        // random row, col in [0..Dim-1]
//        Agt.Row = FMath::RandRange(0, Dim - 1);
//        Agt.Col = FMath::RandRange(0, Dim - 1);
//    }
//
//    // NxN zeros
//    Heights = FMatrix2D(GridSizeY, GridSizeX, 0.0f);
//    PreviousHeights = FMatrix2D(GridSizeY, GridSizeX, 0.0f);
//}
//
//void UDiscreteFourier2D::Reset(int32 NewNumAgents)
//{
//    int32 SafeNum = FMath::Max(NewNumAgents, 0);
//    Agents.SetNum(SafeNum);
//
//    int32 Dim = 2 * K;
//    for (int32 i = 0; i < SafeNum; ++i)
//    {
//        Agents[i].AgentA = FMatrix2D(Dim, Dim, 0.0f);
//        // random row, col
//        Agents[i].Row = FMath::RandRange(0, Dim - 1);
//        Agents[i].Col = FMath::RandRange(0, Dim - 1);
//    }
//
//    // Clear final maps
//    Heights.Init(0.0f);
//    PreviousHeights.Init(0.0f);
//}
//
//const FMatrix2D& UDiscreteFourier2D::Update(const TArray<FAgentFourierDelta>& Deltas)
//{
//    // 1) Apply each agent's discrete move + partial update
//    if (Deltas.Num() != Agents.Num())
//    {
//        UE_LOG(LogTemp, Warning, TEXT("UDiscreteFourier2D::Update: #Deltas != #Agents"));
//    }
//
//    int32 Dim = 2 * K;
//
//    for (int32 i = 0; i < Deltas.Num(); ++i)
//    {
//        if (!Agents.IsValidIndex(i))
//            continue;
//
//        FAgentFourierState& Agt = Agents[i];
//        const FAgentFourierDelta& D = Deltas[i];
//
//        // (A) Move agent's location in row/col with wrap
//        switch (D.Direction)
//        {
//        case EAgentDirection::Up:
//            Agt.Row = (Agt.Row - 1 + Dim) % Dim;
//            break;
//        case EAgentDirection::Down:
//            Agt.Row = (Agt.Row + 1) % Dim;
//            break;
//        case EAgentDirection::Left:
//            Agt.Col = (Agt.Col - 1 + Dim) % Dim;
//            break;
//        case EAgentDirection::Right:
//            Agt.Col = (Agt.Col + 1) % Dim;
//            break;
//        case EAgentDirection::None:
//        default:
//            // no move
//            break;
//        }
//
//        // (B) partial update at (row,col)
//        int32 r = Agt.Row;
//        int32 c = Agt.Col;
//
//        // We'll do increments of ±0.05 or set zero, but then clamp the final cell
//        float oldVal = Agt.AgentA[r][c];
//        float newVal = oldVal;
//
//        switch (D.MatrixUpdate)
//        {
//        case EAgentMatrixUpdate::Inc:
//            newVal += MatrixDeltaRange.Y;
//            break;
//        case EAgentMatrixUpdate::Dec:
//            newVal += MatrixDeltaRange.X;
//            break;
//        case EAgentMatrixUpdate::Zero:
//            newVal = 0.0f;
//            break;
//        case EAgentMatrixUpdate::None:
//        default:
//            // do nothing
//            break;
//        }
//
//        // clamp to MatrixDeltaRange
//        Agt.AgentA[r][c] = newVal;
//    }
//
//    // 2) Sum all agents
//    PreviousHeights = Heights;
//
//    FMatrix2D G_total(GridSizeY, GridSizeX, 0.0f);
//    for (int32 i = 0; i < Agents.Num(); ++i)
//    {
//        FMatrix2D Gi = ComputeAgentHeight(i);
//        G_total += Gi;
//    }
//
//    Heights = G_total;
//
//    return Heights;
//}
//
//const FMatrix2D& UDiscreteFourier2D::GetHeights() const
//{
//    return Heights;
//}
//
//FMatrix2D UDiscreteFourier2D::GetAgentMatrix(int32 AgentIndex) const
//{
//    if (!Agents.IsValidIndex(AgentIndex))
//    {
//        return FMatrix2D(0, 0);
//    }
//    return Agents[AgentIndex].AgentA;
//}
//
//TArray<float> UDiscreteFourier2D::GetAgentFourierState(int32 AgentIndex) const
//{
//    TArray<float> Out;
//    if (!Agents.IsValidIndex(AgentIndex))
//    {
//        return Out;
//    }
//
//    const FAgentFourierState& A = Agents[AgentIndex];
//
//
//    // push row, col, and parameter at loc
//    Out.Add(A.Row);
//    Out.Add(A.Col);
//    Out.Add(A.AgentA[A.Row][A.Col]);
//
//    return Out;
//}
//
///**
// * Sum wave: G_i = BasisSx * AgentA * BasisSy^T
// */
//FMatrix2D UDiscreteFourier2D::ComputeAgentHeight(int32 AgentIndex) const
//{
//    if (!Agents.IsValidIndex(AgentIndex))
//    {
//        return FMatrix2D(GridSizeY, GridSizeX, 0.0f);
//    }
//
//    const FAgentFourierState& Agt = Agents[AgentIndex];
//    FMatrix2D Gtemp = BasisSx.MatMul(Agt.AgentA);
//    FMatrix2D Gi = Gtemp.MatMul(BasisSy.T());
//    return Gi;
//}
//
///**
// * Build the fixed basis Sx => (GridSizeY x 2K), Sy => (GridSizeX x 2K).
// * We do a standard [0..2π) approach, no phase shift or frequency scaling.
// */
//void UDiscreteFourier2D::BuildBasis()
//{
//    // build a helper for (size x K) freq
//    auto MakeFreq = [&](int32 Size, int32 inK) -> FMatrix2D
//        {
//            // (Size x inK) by direct cos/sin? 
//            FMatrix2D CosMat(Size, inK);
//            FMatrix2D SinMat(Size, inK);
//
//            for (int32 i = 0; i < Size; ++i)
//            {
//                float x = 2.f * PI * i / (float)Size;
//                for (int32 m = 0; m < inK; ++m)
//                {
//                    CosMat[i][m] = FMath::Cos(m * x);
//                    SinMat[i][m] = FMath::Sin(m * x);
//                }
//            }
//            // horizontally stack CosMat, SinMat => (Size x 2K)
//            int32 R = Size;
//            int32 C = 2 * inK;
//            FMatrix2D Result(R, C, 0.0f);
//
//            for (int32 rr = 0; rr < R; ++rr)
//            {
//                for (int32 cc = 0; cc < inK; ++cc)
//                {
//                    Result[rr][cc] = CosMat[rr][cc];
//                    Result[rr][inK + cc] = SinMat[rr][cc];
//                }
//            }
//            return Result;
//        };
//
//    BasisSx = MakeFreq(GridSizeY, K); // (GridSizeY x 2K)
//    BasisSy = MakeFreq(GridSizeX, K); // (GridSizeX x 2K)
//}
