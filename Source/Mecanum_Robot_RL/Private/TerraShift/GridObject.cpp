#include "TerraShift/GridObject.h"

AGridObject::AGridObject() {
    PrimaryActorTick.bCanEverTick = false;

    // Create the mesh component
    MeshComponent = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("MeshComponent"));
    RootComponent = MeshComponent;

    // Set default properties
    MeshComponent->SetCollisionEnabled(ECollisionEnabled::QueryOnly);
    MeshComponent->SetCollisionResponseToAllChannels(ECollisionResponse::ECR_Ignore);
    MeshComponent->SetCollisionResponseToChannel(ECollisionChannel::ECC_WorldStatic, ECollisionResponse::ECR_Block);
    MeshComponent->SetMobility(EComponentMobility::Movable);

    bIsActive = false;
}

void AGridObject::InitializeGridObject(FVector InObjectSize) {
    // Set the default mesh (a cube)
    UStaticMesh* DefaultMesh = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Sphere.Sphere"));
    if (DefaultMesh) {
        MeshComponent->SetStaticMesh(DefaultMesh);
        MeshComponent->SetWorldScale3D(InObjectSize);

        // Create and assign a dynamic material instance
        UMaterial* BaseMaterial = LoadObject<UMaterial>(nullptr, TEXT("/Game/StarterContent/Materials/M_Basic_Floor.M_Basic_Floor"));
        if (BaseMaterial) {
            UMaterialInstanceDynamic* DynMaterial = UMaterialInstanceDynamic::Create(BaseMaterial, this);
            MeshComponent->SetMaterial(0, DynMaterial);
        }
    }
    
    SetGridObjectActive(false);
}

void AGridObject::SetGridObjectActive(bool bInIsActive) {
    bIsActive = bInIsActive;
    SetActorHiddenInGame(!bIsActive);
    // MeshComponent->SetSimulatePhysics(bIsActive);
    MeshComponent->SetSimulatePhysics(false);
}

FVector AGridObject::GetObjectExtent() const {
    return MeshComponent->Bounds.BoxExtent * GetActorScale3D();
}

bool AGridObject::IsActive() const {
    return bIsActive;
}
