#include "TerraShift/GridObject.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "UObject/ConstructorHelpers.h"
#include "Engine/StaticMesh.h"
#include "Materials/Material.h"

AGridObject::AGridObject() {
    PrimaryActorTick.bCanEverTick = false;

    // Create the root component (non-simulating)
    GridObjectRoot = CreateDefaultSubobject<USceneComponent>(TEXT("GridObjectRoot"));
    RootComponent = GridObjectRoot;

    // Create the mesh component and attach it to the root
    MeshComponent = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("MeshComponent"));
    MeshComponent->SetupAttachment(GridObjectRoot);

    // Disable physics simulation on the root component
    GridObjectRoot->SetMobility(EComponentMobility::Movable);

    // Set default properties for the mesh component
    MeshComponent->SetCollisionEnabled(ECollisionEnabled::NoCollision);
    MeshComponent->SetMobility(EComponentMobility::Movable);

    // Initialize variables
    bIsActive = false;
    DynMaterial = nullptr;
}

void AGridObject::InitializeGridObject(FVector InObjectSize) {
    // Set the default mesh (a sphere)
    UStaticMesh* DefaultMesh = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Sphere.Sphere"));
    if (DefaultMesh) {
        MeshComponent->SetStaticMesh(DefaultMesh);
        MeshComponent->SetWorldScale3D(InObjectSize);
        MeshComponent->SetRelativeLocation(FVector::ZeroVector);

        // Load and set the dynamic material
        UMaterial* BaseMaterial = LoadObject<UMaterial>(nullptr, TEXT("/Game/StarterContent/Materials/M_Basic_Floor.M_Basic_Floor"));
        if (BaseMaterial) {
            DynMaterial = UMaterialInstanceDynamic::Create(BaseMaterial, this);
            MeshComponent->SetMaterial(0, DynMaterial);
        }

        // Initialize collision settings
        MeshComponent->SetCollisionEnabled(ECollisionEnabled::NoCollision);
        MeshComponent->SetSimulatePhysics(false);
        MeshComponent->SetEnableGravity(false);
    }

    SetGridObjectActive(false);
}

void AGridObject::SetGridObjectActive(bool bInIsActive) {
    bIsActive = bInIsActive;
    SetActorHiddenInGame(!bIsActive);
    SetSimulatePhysics(bIsActive);
}

void AGridObject::SetSimulatePhysics(bool bEnablePhysics) {
    MeshComponent->SetSimulatePhysics(bEnablePhysics);
    MeshComponent->SetEnableGravity(true);

    if (bEnablePhysics) {
        // Enable collision and physics
        MeshComponent->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
        MeshComponent->SetCollisionObjectType(ECollisionChannel::ECC_PhysicsBody);
        MeshComponent->SetCollisionResponseToAllChannels(ECollisionResponse::ECR_Block);

        // Optionally, adjust collision responses
        // MeshComponent->SetCollisionResponseToChannel(ECollisionChannel::ECC_WorldStatic, ECollisionResponse::ECR_Block);

        SetGridObjectColor(FLinearColor::Red);
    }
    else {
        // Disable collision and physics
        MeshComponent->SetCollisionEnabled(ECollisionEnabled::NoCollision);
        SetGridObjectColor(FLinearColor::White);
    }
}

void AGridObject::SetGridObjectColor(FLinearColor Color) {
    if (DynMaterial) {
        DynMaterial->SetVectorParameterValue("Color", Color);
        MeshComponent->SetMaterial(0, DynMaterial);
    }
}

FVector AGridObject::GetObjectExtent() const {
    return MeshComponent->Bounds.BoxExtent * GetActorScale3D();
}

bool AGridObject::IsActive() const {
    return bIsActive;
}
