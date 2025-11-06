#include "TerraShift/MainPlatform.h"

AMainPlatform::AMainPlatform()
{
    PrimaryActorTick.bCanEverTick = false;

    RootSceneComponent = CreateDefaultSubobject<USceneComponent>(TEXT("RootSceneComponent"));
    RootComponent = RootSceneComponent;
    RootSceneComponent->SetMobility(EComponentMobility::Movable);

    PlatformMeshComponent = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("PlatformMeshComponent"));
    PlatformMeshComponent->SetupAttachment(RootComponent);
    PlatformMeshComponent->SetMobility(EComponentMobility::Movable);

    PlatformMeshComponent->SetSimulatePhysics(false);
    PlatformMeshComponent->SetEnableGravity(false);
    PlatformMeshComponent->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
    PlatformMeshComponent->SetCollisionObjectType(ECollisionChannel::ECC_WorldStatic);
    PlatformMeshComponent->SetCollisionResponseToAllChannels(ECollisionResponse::ECR_Block);
}

void AMainPlatform::InitializePlatform(UStaticMesh* Mesh, UMaterial* Material)
{
    if (Mesh)
    {
        PlatformMeshComponent->SetStaticMesh(Mesh);
    }

    if (Material)
    {
        PlatformMeshComponent->SetMaterial(0, Material);
    }
}
