#pragma once

#include "CoreMinimal.h"
#include "Matrix2D.h"

enum class EAgentParameterIndex
{
    PositionX,
    PositionY,
    VelocityX,
    VelocityY,
    Amplitude,
    WaveOrientation,
    Wavenumber,
    Frequency,
    Phase,
    Sigma,
    Time,
    Count
};

struct AgentParameters
{
    FVector2f Position;
    FVector2f Velocity;
    float Amplitude;
    float WaveOrientation;
    float Wavenumber;
    float Frequency;
    float Phase;
    float Sigma;
    float Time;

    TArray<float> ToArray() const
    {
        return {
            Position.X,
            Position.Y,
            Velocity.X,
            Velocity.Y,
            Amplitude,
            WaveOrientation,
            Wavenumber,
            Frequency,
            Phase,
            Sigma,
            Time
        };
    }

    float GetParameter(EAgentParameterIndex Index) const
    {
        switch (Index)
        {
        case EAgentParameterIndex::PositionX: return Position.X;
        case EAgentParameterIndex::PositionY: return Position.Y;
        case EAgentParameterIndex::VelocityX: return Velocity.X;
        case EAgentParameterIndex::VelocityY: return Velocity.Y;
        case EAgentParameterIndex::Amplitude: return Amplitude;
        case EAgentParameterIndex::WaveOrientation: return WaveOrientation;
        case EAgentParameterIndex::Wavenumber: return Wavenumber;
        case EAgentParameterIndex::Frequency: return Frequency;
        case EAgentParameterIndex::Phase: return Phase;
        case EAgentParameterIndex::Sigma: return Sigma;
        case EAgentParameterIndex::Time: return Time;
        case EAgentParameterIndex::Count: return static_cast<int>(EAgentParameterIndex::Count);
        default: return 0.0f;
        }
    }
};

class UNREALRLLABS_API MorletWavelets2D
{
private:
    int32 GridSizeX;
    int32 GridSizeY;
    float PhaseVelocity;

    Matrix2D XGrid;
    Matrix2D YGrid;

    Matrix2D Heights;

    // Clamping range for smooth height transitions
    float MaxDeltaHeight = 0.05f;

public:
    // Constructor
    MorletWavelets2D(int32 InGridSizeX, int32 InGridSizeY, float InPhaseVelocity = 1.0f);

    // Initialize grid coordinates and heights
    void Initialize();

    // Update function with height movement clamping
    Matrix2D Update(const TArray<AgentParameters>& UpdatedParameters);

    // Reset function
    void Reset();

    // Get the current height map
    const Matrix2D& GetHeights() const;

    // Get phase velocity
    float GetPhaseVelocity() const;
};
