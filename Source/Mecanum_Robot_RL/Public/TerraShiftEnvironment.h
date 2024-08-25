#pragma once

#include "CoreMinimal.h"
#include "BaseEnvironment.h"
#include "Engine/StaticMeshActor.h"
#include "RLTypes.h"
#include "TerraShiftEnvironment.generated.h"

// Struct for initialization parameters specific to TerraShiftEnvironment
USTRUCT(BlueprintType)
struct UNREALRLLABS_API FTerraShiftEnvironmentInitParams : public FBaseInitParams
{
    GENERATED_USTRUCT_BODY();

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    float GroundPlaneSize = 2.0f; // m

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    float MaxColumnHeight = 10.0f; // cm

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    FVector ObjectSize = { 0.1f, 0.1f, 0.1f }; // m

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    float ObjectMass = 0.2f; // kg

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    float ColumnMass = 0.01f; // kg

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    float ColumnVelocity = 20.0f; // cm/s

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    float ColumnAccelConstant = 0.2f; // Acceleration constant for columns

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    int GridSize = 20; // Size of the grid

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    int MaxSteps = 1024; // Maximum steps per episode

    // Chute parameters
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Chute Params")
    float ChuteRadius = 0.2f; // Radius of the chute

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Chute Params")
    float ChuteHeight = 0.5f; // Height of the chute

    // Number of goals in the environment
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    int32 NumGoals = 3;
};

// Array of colors for goals
const TArray<FLinearColor> GoalColors = {
        FLinearColor(1.0f, 0.0f, 0.0f),
        FLinearColor(0.0f, 1.0f, 0.0f),
        FLinearColor(0.0f, 0.0f, 1.0f),
        FLinearColor(1.0f, 1.0f, 0.0f),
        FLinearColor(1.0f, 0.0f, 1.0f),
        FLinearColor(0.0f, 1.0f, 1.0f),
        FLinearColor(1.0f, 0.5f, 0.0f),
        FLinearColor(0.5f, 0.0f, 1.0f),
        FLinearColor(1.0f, 0.0f, 0.5f),
        FLinearColor(0.5f, 1.0f, 0.0f),
};

UCLASS()
class UNREALRLLABS_API ATerraShiftEnvironment : public ABaseEnvironment
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

    // The Columns controlled by GridObjects
    UPROPERTY(EditAnywhere)
    TArray<AStaticMeshActor*> Columns;

    // The chute from which objects spawn
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TerraShift")
    AStaticMeshActor* Chute;

    virtual void InitEnv(FBaseInitParams* Params) override;
    virtual FState ResetEnv(int NumAgents) override;
    virtual void Act(FAction Action) override;
    virtual void PostTransition() override;
    virtual void PostStep() override;
    virtual FState State() override;
    virtual bool Done() override;
    virtual bool Trunc() override;
    virtual float Reward() override;

    // Override the Tick function to update column positions
    virtual void Tick(float DeltaTime) override;

private:
    FTerraShiftEnvironmentInitParams* TerraShiftParams = nullptr;

    // State Variables
    int CurrentStep;
    int CurrentAgents;
    TArray<int> LastColumnIndexArray;
    TArray<FVector> GoalPositionArray;
    TArray<int32> AgentGoalIndices;
    TArray<FVector> GridCenterPoints;
    TArray<float> ColumnVelocities;

    // Function to spawn a column in the environment
    AStaticMeshActor* SpawnColumn(FVector Dimensions, FName Name);

    // Function to spawn the ground platform in the environment
    AStaticMeshActor* SpawnPlatform(FVector Location, FVector Size);

    // Function to spawn the chute
    AStaticMeshActor* SpawnChute(FVector Location);

    void SetColumnHeight(int ColumnIndex, float NewHeight);
    void SetColumnAcceleration(int ColumnIndex, float Acceleration);
    AStaticMeshActor* InitializeGridObject();

    // Function to set the color of a column
    void SetColumnColor(int ColumnIndex, FLinearColor Color);

    int SelectColumn(int AgentIndex, int Direction) const;
    TArray<float> AgentGetState(int AgentIndex);
    int Get1DIndexFromPoint(const FIntPoint& point, int gridSize) const;
    float GridDistance(const FIntPoint& Point1, const FIntPoint& Point2) const;
    float Map(float x, float in_min, float in_max, float out_min, float out_max) const;
   
    // Helper function to check if the object is off the platform
    bool ObjectOffPlatform(int AgentIndex) const;

    // Helper function to check if the object has reached the wrong goal
    bool ObjectReachedWrongGoal(int AgentIndex) const;
};