#pragma once

#include "CoreMinimal.h"
#include <deque>
#include <mutex>  // Include for std::mutex
#include "ExperienceBuffer.generated.h"

USTRUCT(BlueprintType)
struct FExperience
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadWrite, Category = "Experience")
        TArray<float> State;

    UPROPERTY(BlueprintReadWrite, Category = "Experience")
        TArray<float> Action;

    UPROPERTY(BlueprintReadWrite, Category = "Experience")
        float Reward;

    UPROPERTY(BlueprintReadWrite, Category = "Experience")
        TArray<float> NextState;

    UPROPERTY(BlueprintReadWrite, Category = "Experience")
        bool Done;
};

USTRUCT(BlueprintType)
struct FExperienceBatch
{
    GENERATED_USTRUCT_BODY()

        UPROPERTY(BlueprintReadWrite, Category = "Experience")
        TArray<FExperience> Experiences;
};

UCLASS(Blueprintable, BlueprintType)
class MECANUM_ROBOT_RL_API UExperienceBuffer : public UObject
{
    GENERATED_BODY()

public:
    UExperienceBuffer();

    void AddExperiences(const TArray<FExperienceBatch>& EnvironmentTrajectories);
    TArray<FExperienceBatch> SampleExperiences(int32 BatchSize);
    void SetBufferCapacity(int32 NewCapacity);

private:
    UPROPERTY()
        int32 BufferCapacity;

    TArray<std::deque<FExperience>> ExperienceDeques;
    std::mutex BufferMutex;

    void EnsureBufferLimit();
};
