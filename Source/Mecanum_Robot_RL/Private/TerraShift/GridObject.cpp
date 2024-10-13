#include "TerraShift/GridObject.h"

AGridObject::AGridObject()
{
    PrimaryActorTick.bCanEverTick = false;

    // Create default subobjects
    MeshComponent = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("MeshComponent"));
    ObjectMesh = CreateDefaultSubobject<UStaticMesh>(TEXT("DefaultMesh"));
    ObjectMaterial = CreateDefaultSubobject<UMaterial>(TEXT("DefaultMaterial"));
    RootComponent = MeshComponent;
}

void AGridObject::InitializeGridObject(FVector InObjectSize, UStaticMesh* Mesh, UMaterial* Material)
{
    // Set mesh and material
    if (Mesh)
    {
        MeshComponent->SetStaticMesh(Mesh);
    }

    if (Material)
    {
        MeshComponent->SetMaterial(0, Material);
    }

    // Set the size and physics for the mesh component
    MeshComponent->SetWorldScale3D(InObjectSize);
    MeshComponent->SetSimulatePhysics(true);
    MeshComponent->SetMobility(EComponentMobility::Movable);
    SetActorHiddenInGame(true);

    ObjectSize = InObjectSize;
}

void AGridObject::SetGridObjectActive(bool SetActive)
{
    SetActorHiddenInGame(!SetActive);
    MeshComponent->SetSimulatePhysics(SetActive);
}

void AGridObject::SetActorLocationAndActivate(FVector NewLocation)
{
    SetGridObjectActive(true);
    SetActorLocation(NewLocation);
}

FVector AGridObject::GetObjectExtent() const
{
    if (MeshComponent && MeshComponent->GetStaticMesh())
    {
        return MeshComponent->GetStaticMesh()->GetBounds().BoxExtent * GetActorScale3D();
    }

    return FVector::ZeroVector;
}
