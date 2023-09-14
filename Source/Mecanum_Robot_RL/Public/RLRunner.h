#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "ExperienceBuffer.h"
#include "BaseEnvironment.h"
#include "VectorEnvironment.h"
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
    void InitRunner(TSubclassOf<ABaseEnvironment> EnvironmentClass, TArray<FBaseInitParams*> ParamsArray, int32 BufferSize, int32 BatchSize);

    // Get actions from the Python model
    TArray<TArray<float>> GetActions(TArray<TArray<float>> State);

    // Train the model
    void Train();

    // Add an experience to the buffer
    void AddExperiences(const TArray<FExperienceBatch>& EnvironmentTrajectories);

    UFUNCTION(BlueprintCallable, Category = "RL|Experience")
        TArray<FExperienceBatch> SampleExperiences(int bSize);

private:
    UPROPERTY()
        AVectorEnvironment* VectorEnvironment;

    UPROPERTY()
        UExperienceBuffer* ExperienceBufferInstance;

    int32 batchSize;
    int32 buffSize;
    TArray<TArray<float>> States;
};
