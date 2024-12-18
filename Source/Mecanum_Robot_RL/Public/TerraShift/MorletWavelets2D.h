#pragma once

#include "CoreMinimal.h"
#include "Matrix2D.h"

enum class EAgentParameterIndex
{
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
    FVector2f Velocity;
    float Amplitude;
    float WaveOrientation;
    float Wavenumber;
    float PhaseVelocity; // Agent-specific phase velocity
    float Phase;
    float Sigma;
    float Time;

    TArray<float> ToArray() const
    {
        return {
            Velocity.X,
            Velocity.Y,
            Amplitude,
            WaveOrientation,
            Wavenumber,
            PhaseVelocity,
            Phase,
            Sigma,
            Time
        };
    }

    float GetParameter(EAgentParameterIndex Index) const
    {
        switch (Index)
        {
        case EAgentParameterIndex::VelocityX: return Velocity.X;
        case EAgentParameterIndex::VelocityY: return Velocity.Y;
        case EAgentParameterIndex::Amplitude: return Amplitude;
        case EAgentParameterIndex::WaveOrientation: return WaveOrientation;
        case EAgentParameterIndex::Wavenumber: return Wavenumber;
        case EAgentParameterIndex::Frequency: return PhaseVelocity;
        case EAgentParameterIndex::Phase: return Phase;
        case EAgentParameterIndex::Sigma: return Sigma;
        case EAgentParameterIndex::Time: return Time;
        default: return 0.0f;
        }
    }
};

class UNREALRLLABS_API MorletWavelets2D
{
private:
    int32 GridSizeX;
    int32 GridSizeY;

    Matrix2D XGrid;
    Matrix2D YGrid;
    Matrix2D Heights;

    // Current positions of agents, initially centered on the grid
    TArray<FVector2f> AgentPositions;

    // Clamping range for smooth height transitions
    float MaxDeltaHeight;

public:
    // Constructor
    MorletWavelets2D(int32 InGridSizeX, int32 InGridSizeY, float MaxDeltaHeight);

    // Initialize grid coordinates and heights
    void Initialize();

    // Update function with position propagation and height movement clamping
    Matrix2D Update(const TArray<AgentParameters>& UpdatedParameters);

    // Reset function to center agents on the grid
    void Reset(int32 NumAgents);

    // Get the current height map
    const Matrix2D& GetHeights() const;

    // Get the current position of a specific agent
    FVector2f GetAgentPosition(int32 AgentIndex) const;
};
