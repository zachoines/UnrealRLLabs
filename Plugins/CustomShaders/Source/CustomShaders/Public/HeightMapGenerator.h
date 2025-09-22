// Minimal API to dispatch GPU height map generation
#pragma once

#include "CoreMinimal.h"

struct CUSTOMSHADERS_API FHeightMapObject
{
    FVector CenterLocal; // in grid local space
    FVector Radii;       // from MeshComponent->Bounds.BoxExtent (world)
};

struct CUSTOMSHADERS_API FHeightMapGenParams
{
    int32 GridSize = 0;
    int32 StateW = 0;
    int32 StateH = 0;
    FVector2D PlatformSize = FVector2D::ZeroVector; // world X,Y size
    float CellSize = 1.f;
    float MinZ = 0.f;
    float MaxZ = 1.f;
    FVector ColumnRadii = FVector::ZeroVector; // world bounds BoxExtent for a column
};

// Returns true on success and fills OutState with StateH*StateW floats in [-1,1]
CUSTOMSHADERS_API bool GenerateHeightMapGPU(const FHeightMapGenParams& Params,
                          const TArray<float>& WaveHeights,
                          const TArray<FHeightMapObject>& Objects,
                          TArray<float>& OutState);


