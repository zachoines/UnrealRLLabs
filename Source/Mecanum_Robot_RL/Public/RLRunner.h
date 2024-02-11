#pragma once

#include "Math/UnrealMathUtility.h"
#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "ExperienceBuffer.h"
#include "VectorEnvironment.h"
#include "SharedMemoryAgentCommunicator.h"
#include "RLTypes.h"
#include "RLRunner.generated.h"


UCLASS(Blueprintable, BlueprintType)
class MECANUM_ROBOT_RL_API ARLRunner : public AActor
{
    GENERATED_BODY()

public:
    // Sets default values for this actor's properties
    ARLRunner();

    // Called every frame
    virtual void Tick(float DeltaTime) override;

    // Initialize the runner
    void InitRunner(
        TSubclassOf<ABaseEnvironment> EnvironmentClass,
        TArray<FBaseInitParams*> ParamsArray,
        FTrainParams TrainParams
    );

    // Get actions from the Python model
    TArray<FAction> GetActions(TArray<FState> States, TArray<float> Dones, TArray<float> Truncs);

    // Add an experience to the buffer
    void AddExperiences(const TArray<FExperienceBatch>& EnvironmentTrajectories);

    TArray<FExperienceBatch> SampleExperiences(int bSize);

    // For multi-agent
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Train Params")
    int CurrentAgents;

private:
    UPROPERTY()
    AVectorEnvironment* VectorEnvironment;

    UPROPERTY()
    UExperienceBuffer* ExperienceBufferInstance;

    UPROPERTY()
    USharedMemoryAgentCommunicator* AgentComm;

    unsigned long int CurrentStep;
    unsigned long int CurrentUpdate;

    FTrainParams TrainerParams;
};
