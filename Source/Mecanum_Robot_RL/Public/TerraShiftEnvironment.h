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
    FVector GroundPlaneSize = FVector::One() * 5.0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    FVector ColumnSize = { 0.25, 0.25, 1.0 };

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    FVector ObjectSize = { 0.25, 0.25, 0.25 };
};


UCLASS()
class MECANUM_ROBOT_RL_API ATerraShiftEnvironment : public ABaseEnvironment
{
    GENERATED_BODY()

public:

    // Sets default values for this actor's properties
    ATerraShiftEnvironment();

    // The columns representing objects
    UPROPERTY(EditAnywhere)
    TArray<AStaticMeshActor*> Columns;

    // The plane that agents operate on
    UPROPERTY(EditAnywhere)
    AStaticMeshActor* GroundPlane;

    // The objects the TerraShift platform moves
    UPROPERTY(EditAnywhere)
    TArray<AStaticMeshActor*> Objects;

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
    FVector GroundPlaneSize;
    FVector ColumnHeadSize;
    FVector GroundPlaneCenter;

    // Constant
    const int GridSize = 100;
    const int MaxSteps = 128;
    const float AgentVisibility = 3;
    const float MaxAgents = 10;
    const float MovementConstraint = 0.1;

    // State Variables
    int CurrentStep;
    int CurrentAgents;
    FTransform GroundPlaneTransform;
    FTransform InverseGroundPlaneTransform;
    TArray<TArray<FVector>> GridCenterPoints;

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
    TMap<AStaticMeshActor*, FIntPoint> ActorToLocationMap;

    TMap<int, TPair<float, float>> ColumnStates;
    TMap<int, TPair<FIntPoint, FIntPoint>> ObjectGoalPositions;

    void MoveAgent(int AgentIndex, float position);
    void SpawnGridObject(FIntPoint SpawnLocation, FIntPoint GaolLocation);
    AStaticMeshActor* InitializeObject(const FLinearColor& Color);
    
    AStaticMeshActor* SpawnColumn(FVector Location, FVector Dimensions);
    AStaticMeshActor* SpawnPlatform(FVector Location, FVector Size);
    void SpawnPlatformWithColumns(FVector CenterPoint, int32 GridSize);
    void AttachPrismaticJoint(AStaticMeshActor* Column, AStaticMeshActor* Platform);
    
    FIntPoint GenerateRandomLocation();
    FVector GetWorldLocationFromGridIndex(FIntPoint GridIndex);

    TArray<float> AgentGetState(int AgentIndex);
    int Get1DIndexFromPoint(const FIntPoint& point, int gridSize);
    float GridDistance(const FIntPoint& Point1, const FIntPoint& Point2);
    float map(float x, float in_min, float in_max, float out_min, float out_max);
};

