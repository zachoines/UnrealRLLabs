#pragma once

#include "CoreMinimal.h"
#include <deque>
#include "BaseEnvironment.h"
#include "ActionSpace.h"
#include "ExperienceBuffer.generated.h"

USTRUCT(BlueprintType)
struct MECANUM_ROBOT_RL_API FExperience
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadWrite, Category = "Experience")
    FState State;

    UPROPERTY(BlueprintReadWrite, Category = "Experience")
    FAction Action;

    UPROPERTY(BlueprintReadWrite, Category = "Experience")
    float Reward;

    UPROPERTY(BlueprintReadWrite, Category = "Experience")
    FState NextState;

    UPROPERTY(BlueprintReadWrite, Category = "Experience")
    bool Trunc;

    UPROPERTY(BlueprintReadWrite, Category = "Experience")
    bool Done;
};

USTRUCT(BlueprintType)
struct MECANUM_ROBOT_RL_API FExperienceBatch
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
    int32 Size();

private:
    UPROPERTY()
    int32 BufferCapacity;

    TArray<TArray<FExperience>> ExperienceDeques;

    void EnsureBufferLimit();
};
