#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "RLTypes.h"
#include "ExperienceBuffer.generated.h"

/**
 * A single experience record: {State, Action, Reward, NextState, Done, Trunc}.
 */
USTRUCT(BlueprintType)
struct FExperience
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadWrite)
    FState State;

    UPROPERTY(BlueprintReadWrite)
    FAction Action;

    UPROPERTY(BlueprintReadWrite)
    float Reward;

    UPROPERTY(BlueprintReadWrite)
    FState NextState;

    UPROPERTY(BlueprintReadWrite)
    bool Trunc;

    UPROPERTY(BlueprintReadWrite)
    bool Done;
};

/**
 * A batch of experiences. Typically from 1 environment if using the
 * 'SampleEnvironmentTrajectories()' approach.
 */
USTRUCT(BlueprintType)
struct FExperienceBatch
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadWrite)
    TArray<FExperience> Experiences;
};

/**
 * UExperienceBuffer stores experiences per-environment in separate "deques".
 *
 * For on-policy PPO, we typically:
 *  - Add experiences per environment step
 *  - Once each environment has at least 'batchSize' experiences, we sample
 *    exactly 'batchSize' from each env => building TArray<FExperienceBatch>
 *    => pass to training.
 */
UCLASS(Blueprintable, BlueprintType)
class UNREALRLLABS_API UExperienceBuffer : public UObject
{
    GENERATED_BODY()

public:
    UExperienceBuffer();

    /**
     * Initialize the buffer with `NumEnvironments`, capacity,
     * plus sampling policy flags (withReplacement, randomSample).
     */
    UFUNCTION(BlueprintCallable)
    void Initialize(int32 InNumEnvs,
        int32 InBufferCapacity,
        bool InSampleWithReplacement,
        bool InRandomSample);

    /** Add a single experience to a specific environment. */
    UFUNCTION(BlueprintCallable)
    void AddExperience(int32 EnvIndex, const FExperience& Exp);

    /**
     * Return how many experiences the environment with the *fewest* experiences has.
     * We use this to see if we can sample a batch from *all* environments.
     */
    UFUNCTION(BlueprintCallable)
    int32 MinSizeAcrossEnvs() const;

    /**
     * Sample exactly `batchSize` experiences from each environment
     * => returns TArray<FExperienceBatch> of size = #envs.
     * If randomSample => pick random items; else pick from front (FIFO).
     * If sampleWithReplacement => do not remove them from the buffer; else remove them.
     */
    UFUNCTION(BlueprintCallable)
    TArray<FExperienceBatch> SampleEnvironmentTrajectories(int32 batchSize);

private:
    UPROPERTY()
    int32 NumEnvironments;

    UPROPERTY()
    int32 BufferCapacity;

    bool bSampleWithReplacement;
    bool bRandomSample;

    /** For each environment => we store experiences in a TArray (like a deque). */
    TArray<TArray<FExperience>> EnvDeques;
};
