using Solia.DataPipeline.Pipeline;

namespace Solia.DataPipeline.Tests.Pipeline;

public class EventClockTests
{
    private static readonly DateTimeOffset T0 = new(2026, 4, 20, 12, 0, 0, TimeSpan.Zero);

    private static EventClock Clock() => new(TimeSpan.FromSeconds(60), confirmations: 3);

    [Fact]
    public void Follows_plausible_events_and_never_goes_backwards()
    {
        EventClock clock = Clock();

        clock.Observe(T0);
        clock.Observe(T0.AddSeconds(10));
        DateTimeOffset afterLate = clock.Observe(T0.AddSeconds(5)); // late, within skew

        Assert.Equal(T0.AddSeconds(10), afterLate);
    }

    [Fact]
    public void Ignores_a_lone_far_future_event()
    {
        EventClock clock = Clock();
        clock.Observe(T0);

        clock.Observe(T0.AddDays(1));
        DateTimeOffset now = clock.Observe(T0.AddSeconds(1));

        Assert.Equal(T0.AddSeconds(1), now);
    }

    [Fact]
    public void Jumps_forward_once_enough_consecutive_events_confirm_it()
    {
        EventClock clock = Clock();
        clock.Observe(T0);

        clock.Observe(T0.AddHours(1).AddSeconds(2));
        clock.Observe(T0.AddHours(1).AddSeconds(1));
        DateTimeOffset now = clock.Observe(T0.AddHours(1).AddSeconds(3));

        Assert.Equal(T0.AddHours(1).AddSeconds(1), now); // the earliest of the confirming run
    }

    [Fact]
    public void A_plausible_event_breaks_a_forward_run()
    {
        EventClock clock = Clock();
        clock.Observe(T0);

        clock.Observe(T0.AddDays(1));
        clock.Observe(T0.AddDays(1));
        clock.Observe(T0.AddSeconds(1)); // run broken
        DateTimeOffset now = clock.Observe(T0.AddDays(1));

        Assert.Equal(T0.AddSeconds(1), now);
    }

    [Fact]
    public void Resets_when_seeded_by_a_bad_future_first_event()
    {
        EventClock clock = Clock();
        clock.Observe(T0.AddDays(1)); // bad first event

        clock.Observe(T0);
        clock.Observe(T0.AddSeconds(2));
        DateTimeOffset now = clock.Observe(T0.AddSeconds(1));

        Assert.Equal(T0.AddSeconds(2), now);
    }
}
