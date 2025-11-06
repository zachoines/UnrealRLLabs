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

/** Batch of experiences sampled from a single environment. */
USTRUCT(BlueprintType)
struct FExperienceBatch
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadWrite)
    TArray<FExperience> Experiences;
};

/** Per-environment experience storage used for on-policy training loops. */
UCLASS(Blueprintable, BlueprintType)
class UNREALRLLABS_API UExperienceBuffer : public UObject
{
    GENERATED_BODY()

public:
    UExperienceBuffer();

    /** Initializes the buffer and sampling policy. */
    UFUNCTION(BlueprintCallable)
    void Initialize(int32 InNumEnvs,
        int32 InBufferCapacity,
        bool InSampleWithReplacement,
        bool InRandomSample);

    /** Add a single experience to a specific environment. */
    UFUNCTION(BlueprintCallable)
    void AddExperience(int32 EnvIndex, const FExperience& Exp);

    /** Returns the minimum experience count over all environments. */
    UFUNCTION(BlueprintCallable)
    int32 MinSizeAcrossEnvs() const;

    /** Samples `batchSize` experiences per environment, respecting the configured policy. */
    UFUNCTION(BlueprintCallable)
    TArray<FExperienceBatch> SampleEnvironmentTrajectories(int32 batchSize);

    /** Clear all stored experiences from every environment. */
    UFUNCTION(BlueprintCallable)
    void Clear();

private:
    UPROPERTY()
    int32 NumEnvironments;

    UPROPERTY()
    int32 BufferCapacity;

    bool bSampleWithReplacement;
    bool bRandomSample;

    /** Per-environment experience queues. */
    TArray<TArray<FExperience>> EnvDeques;
};
