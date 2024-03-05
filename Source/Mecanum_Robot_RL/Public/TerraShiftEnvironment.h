// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "BaseEnvironment.h"
#include "Engine/StaticMeshActor.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "MaterialShared.h"
#include "RLTypes.h"

#include "PhysicsEngine/PhysicsConstraintComponent.h"
#include "Engine/StaticMeshActor.h"
#include "Components/StaticMeshComponent.h"
#include "UObject/ConstructorHelpers.h"
#include "TerraShiftEnvironment.generated.h"

// Derived struct for initialization parameters specific to CubeEnvironment
USTRUCT(BlueprintType)
struct MECANUM_ROBOT_RL_API FTerraShiftEnvironmentInitParams : public FBaseInitParams
{
    GENERATED_USTRUCT_BODY();

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    float GroundPlaneSize = 2.0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    float ColumnHeight = 1;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    FVector ObjectSize = { 0.1, 0.1, 0.1 };

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    float ObjectMass = { 0.2 };

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    float ColumnMass = { 0.0 };
};

UCLASS()
class MECANUM_ROBOT_RL_API ATerraShiftEnvironment : public ABaseEnvironment
{
    GENERATED_BODY()

public:

    ATerraShiftEnvironment();

    // The root component for organizing everything in this environment
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TerraShift")
    USceneComponent* TerraShiftRoot;

    // The plane that agents operate on
    UPROPERTY(EditAnywhere)
    AStaticMeshActor* Platform;

    // The objects the TerraShift platform moves
    UPROPERTY(EditAnywhere)
    TArray<AStaticMeshActor*> Objects;

    // The Columns controlled by agents
    UPROPERTY(EditAnywhere)
    TArray<AStaticMeshActor*> Columns;

    // Constrain joints connecting the columns with platforms
    UPROPERTY(EditAnywhere)
    TArray<UPhysicsConstraintComponent*> PrismaticJoints;

    virtual void InitEnv(FBaseInitParams* Params) override;
    virtual FState ResetEnv(int NumAgents) override;
    virtual void Act(FAction Action) override;
    virtual void PostTransition() override;
    virtual void PostStep() override;
    virtual FState State() override;
    virtual bool Done() override;
    virtual bool Trunc() override;
    virtual float Reward() override;
    void setCurrentAgents(int NumAgents);

private:
    FTerraShiftEnvironmentInitParams* TerraShiftParams = nullptr;

    // Constant
    const int GridSize = 20;
    const int MaxSteps = 1024;
    const float MaxAgents = GridSize * GridSize;
    float CurrentPressure = 1.0;

    // State Variables
    int CurrentStep;
    int CurrentAgents;
    TArray<FVector> GridCenterPoints;
    

    // Function to spawn a column in the environment
    AStaticMeshActor* SpawnColumn(FVector Dimensions, FName Name);

    // Function to spawn the ground platform in the environment
    AStaticMeshActor* SpawnPlatform(FVector Location, FVector Size);

    // Function to attach a prismatic joint between a column and the platform
    UPhysicsConstraintComponent* AttachPrismaticJoint(AStaticMeshActor* Column);


    void SetColumnVelocity(int ColumnIndex, float Velocity);
    void SetColumnHeight(int ColumnIndex, float NewHeight);
    void ApplyForceToColumn(int ColumnIndex, float ForceMagnitude);
    
    const TArray<FLinearColor> Colors = {
        FLinearColor(1.0f, 0.0f, 0.0f),
        FLinearColor(0.0f, 1.0f, 0.0f),
        FLinearColor(0.0f, 0.0f, 1.0f),
        FLinearColor(1.0f, 1.0f, 0.0f),
        FLinearColor(1.0f, 0.0f, 1.0f),
        FLinearColor(0.0f, 1.0f, 1.0f),
        FLinearColor(1.0f, 0.5f, 0.0f),
        FLinearColor(0.5f, 0.0f, 1.0f),
        FLinearColor(1.0f, 0.0f, 0.5f),
        FLinearColor(0.5f, 1.0f, 0.0f)
    };

    
    TMap<FIntPoint, TArray<AStaticMeshActor*>> UsedLocations;

    void MoveAgent(int AgentIndex, float Value);
    void SpawnGridObject(FIntPoint SpawnLocation, FIntPoint GaolLocation);
    AStaticMeshActor* InitializeGridObject(); // (const FLinearColor& Color);
   
    TArray<float> AgentGetState(int AgentIndex);
    int Get1DIndexFromPoint(const FIntPoint& point, int gridSize);
    float GridDistance(const FIntPoint& Point1, const FIntPoint& Point2);
    float map(float x, float in_min, float in_max, float out_min, float out_max);
};

