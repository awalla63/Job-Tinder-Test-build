import { useCallback, useEffect, useRef, useState } from "react";
import { motion, useMotionValue, useTransform, type PanInfo } from "motion/react";

const SWIPE_THRESHOLD = 120; // px of travel before a drag counts as a decision
const VELOCITY_THRESHOLD = 500; // px/s, so a fast flick counts even if it is short

export type Decision = "like" | "pass";

type Props<T> = {
  items: T[];
  getKey: (item: T) => string;
  renderCard: (item: T) => React.ReactNode;
  onDecide: (item: T, decision: Decision) => void;
  /** Human-readable name of the top card, announced to screen readers. */
  describe: (item: T) => string;
  emptyState: React.ReactNode;
};

/**
 * A card stack where drag, keyboard and buttons are three equivalent input paths.
 *
 * This is a desktop web app, so pointer drag can never be the only way to act. Arrow
 * keys and the Pass/Like buttons call exactly the same `decide()` as a drag does, and an
 * aria-live region announces both the new card and the outcome of each decision. The
 * live region is rendered on first paint rather than injected on change -- screen
 * readers ignore regions that appear and populate in the same tick.
 */
export function SwipeDeck<T>({
  items,
  getKey,
  renderCard,
  onDecide,
  describe,
  emptyState,
}: Props<T>) {
  const [index, setIndex] = useState(0);
  const [announcement, setAnnouncement] = useState("");
  const cardRef = useRef<HTMLDivElement>(null);

  const x = useMotionValue(0);
  const rotate = useTransform(x, [-300, 0, 300], [-14, 0, 14]);
  const likeOpacity = useTransform(x, [40, 160], [0, 1]);
  const passOpacity = useTransform(x, [-160, -40], [1, 0]);

  const current = items[index];
  const remaining = items.length - index;

  const decide = useCallback(
    (decision: Decision) => {
      const item = items[index];
      if (!item) return;
      setAnnouncement(
        `${decision === "like" ? "Liked" : "Passed"} ${describe(item)}. ${
          remaining - 1 > 0 ? `${remaining - 1} cards left.` : "No cards left."
        }`
      );
      onDecide(item, decision);
      x.set(0);
      setIndex((i) => i + 1);
    },
    [items, index, onDecide, describe, remaining, x]
  );

  // Reset when a fresh deck arrives.
  useEffect(() => {
    setIndex(0);
    x.set(0);
  }, [items, x]);

  // Announce each new card, and move focus to it so keyboard users keep their place
  // instead of being dropped onto <body> as the stack advances.
  useEffect(() => {
    if (!current) return;
    setAnnouncement(`Showing ${describe(current)}. ${remaining} cards left.`);
    cardRef.current?.focus();
  }, [index, current, describe, remaining]);

  const onKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      decide("pass");
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      decide("like");
    }
  };

  const onDragEnd = (_: unknown, info: PanInfo) => {
    if (info.offset.x > SWIPE_THRESHOLD || info.velocity.x > VELOCITY_THRESHOLD) {
      decide("like");
    } else if (info.offset.x < -SWIPE_THRESHOLD || info.velocity.x < -VELOCITY_THRESHOLD) {
      decide("pass");
    }
    // Otherwise motion springs the card back to centre on its own.
  };

  return (
    <div className="flex flex-col items-center gap-6">
      {/* Present from first paint so announcements are reliably picked up. */}
      <div aria-live="polite" aria-atomic="true" className="sr-only">
        {announcement}
      </div>

      <div className="relative h-[30rem] w-full max-w-xl">
        {!current && (
          <div className="flex h-full items-center justify-center rounded-2xl border border-dashed border-slate-300 bg-white/60 p-8 text-center">
            {emptyState}
          </div>
        )}

        {/* The card behind the top one, for depth. Inert and hidden from AT. */}
        {items[index + 1] && (
          <div
            aria-hidden="true"
            className="absolute inset-0 scale-95 rounded-2xl border border-slate-200 bg-white opacity-60 shadow-sm"
          />
        )}

        {current && (
          <motion.div
            key={getKey(current)}
            ref={cardRef}
            role="group"
            tabIndex={0}
            aria-label={describe(current)}
            onKeyDown={onKeyDown}
            drag="x"
            dragConstraints={{ left: 0, right: 0 }}
            dragElastic={0.6}
            onDragEnd={onDragEnd}
            style={{ x, rotate }}
            initial={{ scale: 0.97, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="absolute inset-0 cursor-grab overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-lg outline-none focus-visible:ring-4 focus-visible:ring-indigo-300 active:cursor-grabbing motion-reduce:transition-none"
          >
            <motion.div
              style={{ opacity: likeOpacity }}
              aria-hidden="true"
              className="pointer-events-none absolute right-5 top-5 z-10 rounded-lg border-4 border-emerald-500 px-3 py-1 text-xl font-black uppercase tracking-wider text-emerald-500"
            >
              Like
            </motion.div>
            <motion.div
              style={{ opacity: passOpacity }}
              aria-hidden="true"
              className="pointer-events-none absolute left-5 top-5 z-10 rounded-lg border-4 border-rose-500 px-3 py-1 text-xl font-black uppercase tracking-wider text-rose-500"
            >
              Pass
            </motion.div>
            {renderCard(current)}
          </motion.div>
        )}
      </div>

      <div className="flex items-center gap-4">
        <button
          type="button"
          onClick={() => decide("pass")}
          disabled={!current}
          className="rounded-full border-2 border-rose-200 bg-white px-7 py-3 font-semibold text-rose-600 shadow-sm transition hover:bg-rose-50 focus-visible:ring-4 focus-visible:ring-rose-200 disabled:opacity-40"
        >
          Pass
        </button>
        <p className="w-28 text-center text-sm text-slate-500" aria-hidden="true">
          {remaining > 0 ? `${remaining} left` : "—"}
        </p>
        <button
          type="button"
          onClick={() => decide("like")}
          disabled={!current}
          className="rounded-full border-2 border-emerald-200 bg-white px-7 py-3 font-semibold text-emerald-600 shadow-sm transition hover:bg-emerald-50 focus-visible:ring-4 focus-visible:ring-emerald-200 disabled:opacity-40"
        >
          Like
        </button>
      </div>

      <p className="text-xs text-slate-400">
        Drag the card, use the buttons, or press ← / → on your keyboard.
      </p>
    </div>
  );
}
