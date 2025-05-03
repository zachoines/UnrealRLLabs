// NOTICE: This file includes modifications generated with the assistance of generative AI.
// Original code structure and logic by the project author.

#include "TerraShift/OccupancyGrid.h"
#include "TerraShift/Matrix2D.h"
#include "Engine/World.h"

// Sample skeleton
void UOccupancyGrid::InitGrid(int32 InGridSize, float InPlatformSize, FVector InPlatformCenter)
{
    GridSize = InGridSize;
    PlatformSize = InPlatformSize;
    PlatformCenter = InPlatformCenter;
    CellSize = (GridSize > 0) ? (PlatformSize / (float)GridSize) : 1.f;

    // Clear existing layers
    Layers.Empty();
}

void UOccupancyGrid::ResetGrid()
{
    Layers.Empty();
}

int32 UOccupancyGrid::AddObjectToGrid(
    int32 ObjectId,
    FName LayerName,
    float Radius,
    const TArray<FName>& OverlapLayers
)
{
    // 1) find a free cell index => for example:
    //    int32 chosenCell = FindFreeCellForRadius(Radius, OverlapLayers);
    //    if (chosenCell == -1) => fail
    // 2) Then compute the OccupiedCells
    //    TSet<int32> occ = ComputeOccupiedCells(chosenCell, RadiusCells)
    // 3) Store in the layer => create layer if needed

    if (!Layers.Contains(LayerName))
    {
        FOccupancyLayer newLayer;
        Layers.Add(LayerName, newLayer);
    }

    int32 chosenCell = FindFreeCellForRadius(Radius, OverlapLayers);
    if (chosenCell < 0)
    {
        return -1;
    }

    FOccupancyLayer& layer = Layers[LayerName];
    FOccupancyNode newNode;
    newNode.ObjectId = ObjectId;
    newNode.Radius = Radius;

    TSet<int32> occ = ComputeOccupiedCells(chosenCell, Radius);
    newNode.OccupiedCells = occ;

    layer.Objects.Add(ObjectId, newNode);

    return chosenCell;
}

void UOccupancyGrid::RemoveObject(int32 ObjectId, FName LayerName)
{
    if (Layers.Contains(LayerName))
    {
        FOccupancyLayer& layer = Layers[LayerName];
        layer.Objects.Remove(ObjectId);
        // Freed those cells from that layer
    }
}

void UOccupancyGrid::UpdateObject(int32 ObjectId, FName LayerName, float NewRadius, const TArray<FName>& OverlapLayers)
{
    if (!Layers.Contains(LayerName)) return;

    FOccupancyLayer& layer = Layers[LayerName];
    if (!layer.Objects.Contains(ObjectId)) return;

    // 1) Remove old occupied cells from our data structure
    FOccupancyNode& node = layer.Objects[ObjectId];
    node.OccupiedCells.Empty();

    // 2) find a new cell (maybe we do a location param or some logic)
    int32 chosenCell = FindFreeCellForRadius(NewRadius, OverlapLayers);
    if (chosenCell < 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("UpdateObject => cannot find new location => not updated."));
        return;
    }

    node.Radius = NewRadius;
    TSet<int32> newOcc = ComputeOccupiedCells(chosenCell, NewRadius);
    node.OccupiedCells = newOcc;
}

FMatrix2D UOccupancyGrid::GetOccupancyMatrix(const TArray<FName>& LayersToUse, bool bUseBinary) const
{
    FMatrix2D mat(GridSize, GridSize, -1.f); // -1 => free

    // For each layer in LayersToUse => fill
    for (FName layerName : LayersToUse)
    {
        if (!Layers.Contains(layerName)) continue;
        const FOccupancyLayer& layer = Layers[layerName];

        // for each object => for each cell => fill
        for (auto& kv : layer.Objects)
        {
            const FOccupancyNode& node = kv.Value;
            int32 id = node.ObjectId;

            for (int32 cellIdx : node.OccupiedCells)
            {
                int x = cellIdx / GridSize;   // or your transform from 1D to 2D
                int y = cellIdx % GridSize;

                if (bUseBinary)
                {
                    mat[x][y] = 1.f;
                }
                else
                {
                    // store the object id
                    mat[x][y] = float(id);
                }
            }
        }
    }
    return mat;
}

FVector UOccupancyGrid::GridToWorld(int32 GridIndex) const
{
    // e.g. get X,Y
    int32 x = GridIndex / GridSize;
    int32 y = GridIndex % GridSize;

    float half = PlatformSize * 0.5f;
    float worldX = (x + 0.5f) * CellSize - half + PlatformCenter.X;
    float worldY = (y + 0.5f) * CellSize - half + PlatformCenter.Y;
    float worldZ = PlatformCenter.Z; // or top of column => depends on your logic
    return FVector(worldX, worldY, worldZ);
}

int32 UOccupancyGrid::WorldToGrid(const FVector& WorldLocation) const
{
    float half = PlatformSize * 0.5f;

    // convert to local
    float lx = WorldLocation.X - (PlatformCenter.X - half);
    float ly = WorldLocation.Y - (PlatformCenter.Y - half);

    int gx = FMath::FloorToInt(lx / CellSize);
    int gy = FMath::FloorToInt(ly / CellSize);

    // clamp
    gx = FMath::Clamp(gx, 0, GridSize - 1);
    gy = FMath::Clamp(gy, 0, GridSize - 1);

    return gx * GridSize + gy;
}

bool UOccupancyGrid::WouldOverlap(int32 ProposedCellIndex, float RadiusCells, const TArray<FName>& OverlapLayers) const
{
    // 1) figure out which cells the new object would occupy => 
    TSet<int32> testCells = ComputeOccupiedCells(ProposedCellIndex, RadiusCells);

    // 2) for each layer => if it's not in OverlapLayers => check collision
    for (auto& layerKV : Layers)
    {
        FName ln = layerKV.Key;
        // skip if OverlapLayers contains it => meaning we ALLOW overlap
        if (OverlapLayers.Contains(ln))
        {
            continue;
        }
        // otherwise => we do not allow overlap
        const FOccupancyLayer& lay = layerKV.Value;
        // for each object in that layer => if there's ANY intersection => return true
        for (auto& objKV : lay.Objects)
        {
            const FOccupancyNode& node = objKV.Value;
            // check if there's intersection between testCells and node.OccupiedCells
            for (int32 c : testCells)
            {
                if (node.OccupiedCells.Contains(c))
                {
                    return true; // yes => overlap
                }
            }
        }
    }

    return false; // no overlap found
}

int32 UOccupancyGrid::FindFreeCellForRadius(float RadiusCells, const TArray<FName>& OverlapLayers)
{
    static const int32 MaxAttempts = 30;
    int32 totalCells = GridSize * GridSize;

    for (int32 attempt = 0; attempt < MaxAttempts; attempt++)
    {
        // 1) Pick a random cell
        int32 tryIndex = FMath::RandRange(0, totalCells - 1);

        // 2) Check if occupant is fully within grid bounds
        int32 gx = tryIndex / GridSize;
        int32 gy = tryIndex % GridSize;
        int32 R = FMath::CeilToInt(RadiusCells);

        // If occupant circle would go out of [0, GridSize-1], skip
        if ((gx - R) < 0 || (gx + R) >= GridSize || (gy - R) < 0 || (gy + R) >= GridSize)
        {
            continue; // occupant would partially lie outside the grid => skip
        }

        // 3) Check overlap with layers
        bool bOverlap = WouldOverlap(tryIndex, RadiusCells, OverlapLayers);
        if (!bOverlap)
        {
            // Success => occupant can be placed here
            return tryIndex;
        }
    }

    return -1; // no free cell found
}

TSet<int32> UOccupancyGrid::ComputeOccupiedCells(int32 CellIndex, float RadiusCells) const
{
    // For a circle of radius => gather which cellIndices are inside
    // Convert 1D => (gx,gy)
    int gx = CellIndex / GridSize;
    int gy = CellIndex % GridSize;

    TSet<int32> outSet;

    // We'll approximate: we do a bounding box around [gx - R, gx + R], etc.
    int32 R = FMath::CeilToInt(RadiusCells);
    int32 minX = FMath::Max(gx - R, 0);
    int32 maxX = FMath::Min(gx + R, GridSize - 1);
    int32 minY = FMath::Max(gy - R, 0);
    int32 maxY = FMath::Min(gy + R, GridSize - 1);

    for (int32 xx = minX; xx <= maxX; xx++)
    {
        for (int32 yy = minY; yy <= maxY; yy++)
        {
            // check distance to center
            float dx = (xx - gx);
            float dy = (yy - gy);
            float dist2 = dx * dx + dy * dy;
            if (dist2 <= RadiusCells * RadiusCells)
            {
                int32 cIndex = xx * GridSize + yy;
                outSet.Add(cIndex);
            }
        }
    }

    return outSet;
}

TSet<int32> UOccupancyGrid::GetCellOccupants(FName LayerName, int32 CellIndex) const
{
    TSet<int32> outSet;
    if (!Layers.Contains(LayerName))
        return outSet; // empty

    const FOccupancyLayer& layer = Layers[LayerName];

    // For each occupant => if its OccupiedCells contains CellIndex => occupant is in that cell
    for (auto& kv : layer.Objects)
    {
        int32 occupantId = kv.Key;
        const FOccupancyNode& node = kv.Value;

        if (node.OccupiedCells.Contains(CellIndex))
        {
            outSet.Add(occupantId);
        }
    }
    return outSet;
}

bool UOccupancyGrid::GetOccupancyNode(FName LayerName, int32 OccupantId, FOccupancyNode& OutNode) const
{
    // check if layer exists
    if (!Layers.Contains(LayerName))
    {
        return false;
    }
    const FOccupancyLayer& layer = Layers[LayerName];

    // check if occupant exists in that layer
    if (!layer.Objects.Contains(OccupantId))
    {
        return false;
    }

    OutNode = layer.Objects[OccupantId];
    return true;
}

// Example signature
void UOccupancyGrid::UpdateObjectPosition(
    int32 ObjectId,
    FName LayerName,
    int32 DesiredCellIndex,
    float RadiusCells,
    const TArray<FName>& OverlapLayers
)
{
    // 1) Access the layer
    if (!Layers.Contains(LayerName))
    {
        // create if needed or early return
        Layers.Add(LayerName, FOccupancyLayer());
    }
    FOccupancyLayer& layer = Layers[LayerName];

    // 2) If object does not exist, create a new FOccupancyNode
    if (!layer.Objects.Contains(ObjectId))
    {
        FOccupancyNode newNode;
        newNode.ObjectId = ObjectId;
        layer.Objects.Add(ObjectId, newNode);
    }

    // 3) Clear old OccupiedCells
    FOccupancyNode& node = layer.Objects[ObjectId];
    node.OccupiedCells.Empty();

    // 4) Compute new occupied cells
    node.Radius = RadiusCells;
    TSet<int32> newOcc = ComputeOccupiedCells(DesiredCellIndex, RadiusCells);

    // 5) If you want to forbid overlaps with other layers, do a quick check 
    //    but typically you'd allow it or do that outside this function

    node.OccupiedCells = newOcc;
}
