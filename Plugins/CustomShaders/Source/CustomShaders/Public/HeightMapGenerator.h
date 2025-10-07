// Minimal API to dispatch GPU height map generation
#pragma once

#include "CoreMinimal.h"

class FRHIGPUBufferReadback;

struct CUSTOMSHADERS_API FHeightMapGPUDispatchHandle
{
    TSharedPtr<FRHIGPUBufferReadback, ESPMode::ThreadSafe> Readback;
    uint64 ExpectedBytes = 0;
    int32 ElementCount = 0;
    bool bInFlight = false;

    bool IsActive() const { return bInFlight; }
    void Reset()
    {
        bInFlight = false;
        ExpectedBytes = 0;
        ElementCount = 0;
    }
};

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
    FVector ColumnRadii = FVector::ZeroVector; // legacy (unused when per-column radii provided)
    float ColZBias = 0.0f;  // additive Z bias for columns (world units in grid-local)
    float ObjZBias = -0.1f;  // additive Z bias for objects (world units in grid-local)
    int32 NumObjects = 0;
};

// Dispatch/resolve helpers for async height map generation
CUSTOMSHADERS_API bool DispatchHeightMapGPU(
                          const FHeightMapGenParams& Params,
                          const TArray<FVector3f>& ColumnCenters,
                          const TArray<FVector3f>& ColumnRadiiArray,
                          const TArray<FVector3f>& ObjectCenters,
                          const TArray<FVector3f>& ObjectRadii,
                          FHeightMapGPUDispatchHandle& InOutHandle);

CUSTOMSHADERS_API bool ResolveHeightMapGPU(FHeightMapGPUDispatchHandle& Handle, TArray<float>& OutState);

// Returns true on success and fills OutState with StateH*StateW floats in [-1,1]
CUSTOMSHADERS_API bool GenerateHeightMapGPU(
                          const FHeightMapGenParams& Params,
                          const TArray<FVector>& ColumnCenters,
                          const TArray<FVector>& ColumnRadiiArray,
                          const TArray<FHeightMapObject>& Objects,
                          TArray<float>& OutState);
