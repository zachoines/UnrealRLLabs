#include "TerraShift/Grid.h"

AGrid::AGrid() {
    PrimaryActorTick.bCanEverTick = false;
    RootComponent = CreateDefaultSubobject<USceneComponent>(TEXT("GridRoot"));
}

void AGrid::InitializeGrid(int32 InGridSize, float InPlatformSize, FVector Location) {
    GridSize = InGridSize;
    PlatformSize = InPlatformSize;
    CellSize = PlatformSize / static_cast<float>(GridSize);

    if (UWorld* World = GetWorld()) {
        for (int32 X = 0; X < GridSize; ++X) {
            for (int32 Y = 0; Y < GridSize; ++Y) {
                FVector ColumnLocation = CalculateColumnLocation(X, Y, 0.0f);
                AColumn* NewColumn = World->SpawnActor<AColumn>(AColumn::StaticClass(), ColumnLocation, FRotator::ZeroRotator);
                if (NewColumn) {
                    // Attach column to the Grid's root component
                    NewColumn->AttachToComponent(RootComponent, FAttachmentTransformRules::KeepRelativeTransform);
                    NewColumn->InitColumn(FVector(1.0 / GridSize, 1.0 / GridSize, 2.0 / GridSize), ColumnLocation);
                    Columns.Add(NewColumn);
                }
            }
        }
    }

    SetActorLocation(Location);
    ResetGrid();
}

void AGrid::UpdateColumnHeights(const FMatrix2D& HeightMap) {
    for (int32 X = 0; X < GridSize; ++X) {
        for (int32 Y = 0; Y < GridSize; ++Y) {
            int32 Index = X * GridSize + Y;
            if (Columns.IsValidIndex(Index)) {
                Columns[Index]->SetColumnHeight(HeightMap[X][Y]);
            }
        }
    }
}

void AGrid::TogglePhysicsForColumns(const TArray<int32>& ColumnIndices, const TArray<bool>& EnablePhysics) {
    if (ColumnIndices.Num() != EnablePhysics.Num()) {
        UE_LOG(LogTemp, Warning, TEXT("Mismatch in column indices and enable physics arrays."));
        return;
    }

    for (int32 i = 0; i < ColumnIndices.Num(); ++i) {
        if (Columns.IsValidIndex(ColumnIndices[i])) {
            SetColumnPhysics(ColumnIndices[i], EnablePhysics[i]);
        }
    }
}

void AGrid::ResetGrid() {
    for (AColumn* Column : Columns) {
        if (Column) {
            Column->ResetColumn(0.0);
        }
    }
}

FVector AGrid::GetColumnWorldLocation(int32 ColumnIndex) const {
    if (Columns.IsValidIndex(ColumnIndex)) {
        return Columns[ColumnIndex]->GetActorLocation();
    }
    return FVector::ZeroVector;
}

TArray<FVector> AGrid::GetColumnCenters() const {
    TArray<FVector> ColumnCenters;
    for (const AColumn* Column : Columns) {
        if (Column) {
            ColumnCenters.Add(Column->GetActorLocation());
        }
    }
    return ColumnCenters;
}

FVector AGrid::CalculateColumnLocation(int32 X, int32 Y, float Height) const {
    float HalfPlatformSize = PlatformSize / 2.0f;
    float LocationX = (X * CellSize) - HalfPlatformSize + (CellSize / 2.0f);
    float LocationY = (Y * CellSize) - HalfPlatformSize + (CellSize / 2.0f);
    return FVector(LocationX, LocationY, Height);
}

FIntPoint AGrid::Get2DIndexFrom1D(int32 Index) const {
    return FIntPoint(Index / GridSize, Index % GridSize);
}

void AGrid::SetColumnPhysics(int32 ColumnIndex, bool bEnablePhysics) {
    if (Columns.IsValidIndex(ColumnIndex)) {
        Columns[ColumnIndex]->SetSimulatePhysics(bEnablePhysics);
    }
}

void AGrid::SetColumnMovementBounds(float Min, float Max) {
    MinHeight = Min;
    MaxHeight = Max;
}

float AGrid::Map(float x, float in_min, float in_max, float out_min, float out_max) {
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

FVector AGrid::GetColumnOffsets(int32 X, int32 Y) const {
    int32 Index = X * GridSize + Y;
    if (Columns.IsValidIndex(Index)) {
        const AColumn* Column = Columns[Index];
        if (Column && Column->ColumnMesh) {

            // Calculate the local bounds of the column mesh
            FBoxSphereBounds ColumnBounds = Column->ColumnMesh->CalcLocalBounds();

            // Get the local top offset accounting for scaling
            FVector LocalTopOffset = ColumnBounds.BoxExtent * Column->ColumnMesh->GetRelativeScale3D();

            // Add the column's current height to get the total local Z-offset
            return LocalTopOffset + FVector(0.0f, 0.0f, Column->GetColumnHeight());
        }
    }
    return FVector::ZeroVector;
}

FVector2D AGrid::CalculateEdgeCorrectiveOffsets(int32 X, int32 Y) const {
    int32 Index = X * GridSize + Y;
    if (!Columns.IsValidIndex(Index)) {
        return FVector2D::ZeroVector; // Safety check
    }

    // Get the column's local bounds
    const AColumn* Column = Columns[Index];
    if (Column && Column->ColumnMesh) {
        FBoxSphereBounds ColumnBounds = Column->ColumnMesh->CalcLocalBounds();
        float ColumnWidthX = ColumnBounds.BoxExtent.X * Column->ColumnMesh->GetRelativeScale3D().X;
        float ColumnWidthY = ColumnBounds.BoxExtent.Y * Column->ColumnMesh->GetRelativeScale3D().Y;

        // Calculate the corrective offsets based on the column size
        float CorrectiveXOffset = FMath::Min(CellSize / 2.0f, ColumnWidthX);
        float CorrectiveYOffset = FMath::Min(CellSize / 2.0f, ColumnWidthY);

        return FVector2D(CorrectiveXOffset, CorrectiveYOffset);
    }

    return FVector2D::ZeroVector;
}

int32 AGrid::GetTotalColumns() const {
    return Columns.Num();
}

float AGrid::GetColumnHeight(int32 ColumnIndex) const {
    if (Columns.IsValidIndex(ColumnIndex)) {
        return Columns[ColumnIndex]->GetColumnHeight();
    }
    return 0.0f;
}

void AGrid::SetColumnColor(int32 ColumnIndex, const FLinearColor& Color) {
    if (Columns.IsValidIndex(ColumnIndex)) {
        Columns[ColumnIndex]->SetColumnColor(Color);
    }
}

float AGrid::GetMinHeight() const {
    return MinHeight;
}

float AGrid::GetMaxHeight() const {
    return MaxHeight;
}