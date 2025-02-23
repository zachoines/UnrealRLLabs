#include "ExperienceBuffer.h"

UExperienceBuffer::UExperienceBuffer()
    : NumEnvironments(0)
    , BufferCapacity(0)
    , bSampleWithReplacement(false)
    , bRandomSample(false)
{
}

void UExperienceBuffer::Initialize(int32 InNumEnvs,
    int32 InBufferCapacity,
    bool InSampleWithReplacement,
    bool InRandomSample)
{
    NumEnvironments = InNumEnvs;
    BufferCapacity = InBufferCapacity;
    bSampleWithReplacement = InSampleWithReplacement;
    bRandomSample = InRandomSample;

    EnvDeques.SetNum(NumEnvironments);
}

void UExperienceBuffer::AddExperience(int32 EnvIndex, const FExperience& Exp)
{
    if (EnvIndex < 0 || EnvIndex >= EnvDeques.Num())
    {
        UE_LOG(LogTemp, Warning, TEXT("AddExperience: invalid EnvIndex=%d"), EnvIndex);
        return;
    }

    // Append
    EnvDeques[EnvIndex].Add(Exp);

    // If over capacity => remove oldest
    while (EnvDeques[EnvIndex].Num() > BufferCapacity)
    {
        EnvDeques[EnvIndex].RemoveAt(0, 1, false);
    }
}

int32 UExperienceBuffer::MinSizeAcrossEnvs() const
{
    if (EnvDeques.Num() == 0) return 0;
    int32 minVal = INT_MAX;
    for (const auto& dq : EnvDeques)
    {
        if (dq.Num() < minVal)
        {
            minVal = dq.Num();
        }
    }
    return (minVal == INT_MAX ? 0 : minVal);
}

TArray<FExperienceBatch> UExperienceBuffer::SampleEnvironmentTrajectories(int32 batchSize)
{
    TArray<FExperienceBatch> out;
    out.SetNum(EnvDeques.Num());

    for (int32 e = 0; e < EnvDeques.Num(); e++)
    {
        FExperienceBatch batch;

        if (EnvDeques[e].Num() < batchSize)
        {
            // Not enough data => return empty
            // or you can handle partial
            UE_LOG(LogTemp, Warning, TEXT("Env=%d has only %d experiences < batchSize=%d"), e, EnvDeques[e].Num(), batchSize);
            return TArray<FExperienceBatch>();
        }

        if (bRandomSample)
        {
            // sample random indices
            for (int32 i = 0; i < batchSize; i++)
            {
                int32 idx = FMath::RandRange(0, EnvDeques[e].Num() - 1);
                batch.Experiences.Add(EnvDeques[e][idx]);

                if (!bSampleWithReplacement)
                {
                    EnvDeques[e].RemoveAt(idx, 1, false);
                }
            }
        }
        else
        {
            // chronological => from front of the deque
            for (int32 i = 0; i < batchSize; i++)
            {
                batch.Experiences.Add(EnvDeques[e][0]);
                if (!bSampleWithReplacement)
                {
                    EnvDeques[e].RemoveAt(0, 1, false);
                }
            }
        }

        out[e] = batch;
    }

    return out;
}