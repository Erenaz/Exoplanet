#!/usr/bin/env python3
"""
Improved Mini realistic exoplanet ML pipeline (Kepler)
Better detrending + more robust BLS period search
Preserves original crash-safety and comments where possible

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from astropy.timeseries import BoxLeastSquares
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from wotan import flatten
import lightkurve as lk

# -----------------------------
# Configuration
# -----------------------------

OUTPUT_PDF = "kepler_process_improved.pdf"
N_TARGETS = 500
RANDOM_SEED = 51

np.random.seed(RANDOM_SEED)

# -----------------------------
# Helper functions
# -----------------------------

def save_candidate_npz(base_dir, kepid, phase, flux_folded, meta):
    os.makedirs(base_dir, exist_ok=True)
    fname = os.path.join(base_dir, f"candidatev2_{kepid}.npz")
    np.savez(
        fname,
        kepid=int(kepid),
        phase=phase.astype(np.float32),
        flux=flux_folded.astype(np.float32),
        period=float(meta["period"]),
        t0=float(meta["t0"]),
        duration=float(meta["duration"]),
        depth=float(meta["depth"]),
    )
    print(" - Wrote candidate to ", fname)
    return fname


def clean_lc(lc):
    """Initial cleaning: remove NaNs and strong outliers (but preserve transits)"""
    return lc.remove_nans().remove_outliers(sigma=7).normalize()


def robust_flatten(time, flux, window_length=2.0):
    """Multi-pass biweight flattening to remove stellar variability"""
    # First pass: long window to remove low-frequency trends
    print (" - flatten pass 1")
    flat_flux, trend1 = flatten(time, flux, method='mean', window_length=window_length*0.4, break_tolerance=0.5, return_trend=True)
    # Second pass: slightly shorter window to clean residuals
    flat_flux, trend2 = flatten(time, flat_flux, method='biweight', window_length=window_length*0.7, return_trend=True)
    print (" - flatten pass 2")
    final_trend = trend1 * trend2
    return flat_flux, final_trend


def plot_subsample(ax, time, flux, trend, frac=0.01):
    n = len(time)
    k = max(1, int(frac * n))
    idx = np.random.choice(n, k, replace=False)
    ax.plot(time[idx], flux[idx], ".", color='gray', alpha=0.4, markersize=2, label='flux')
    ax.plot(time[idx], trend[idx], ".", color='orange', alpha=0.6, markersize=2, label='trend')


def folded_views(time, flux, period, t0, nbins=201):
    phase = ((time - t0 + 0.5 * period) % period) / period
    order = np.argsort(phase)
    phase, flux = phase[order], flux[order]

    bins = np.linspace(0.0, 1.0, nbins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    digitized = np.digitize(phase, bins) - 1
    binned_flux = np.array([
        np.mean(flux[digitized == i]) if np.any(digitized == i) else np.nan
        for i in range(nbins)
    ])
    # Fill empty bins with median (ML-safe)
    mask = np.isnan(binned_flux)
    if np.any(mask):
        binned_flux[mask] = np.nanmedian(binned_flux)

    return bin_centers, binned_flux


def refine_period(time, flux, coarse_period, coarse_t0):
    periods = np.linspace(coarse_period * 0.95, coarse_period * 1.05, 500)
    durations = np.linspace(0.02, 0.3, 8)
    bls = BoxLeastSquares(time, flux)
    result = bls.power(periods, durations)
    best_idx = np.argmax(result.power)
    return (
        result.period[best_idx],
        result.transit_time[best_idx],
        result.duration[best_idx],
        result.depth[best_idx],
    )


# -----------------------------
# Load KOI catalog; we need this to get the true labels 
# -----------------------------

filename = "koi_catalog.csv"
if not os.path.exists(filename):
    print("Downloading KOI catalog...")
    koi = NasaExoplanetArchive.query_criteria(
        table="cumulative",
        where="koi_disposition IN ('CONFIRMED', 'FALSE POSITIVE')"
    ).to_pandas()
    koi.to_csv(filename, index=False)
else:
    print("Loading KOI catalog from local cache...")
    koi = pd.read_csv(filename)

# drop any duplicates
koi = koi.drop_duplicates(subset="kepid", keep='first')


# -----------------------------
# Choose which stars to process
# -----------------------------

# set to False to reprocess existing FITS files
# set to True to download new stars and process 
randomStars=True

from pathlib import Path
if (randomStars):
    print (" choosing ",N_TARGETS, " random stars")
    koi = koi.sample(N_TARGETS, random_state=RANDOM_SEED)
else:
    ## what FITS files do we have?
    fits_dir = Path("fits")
    kids = [    int(p.stem.split("_")[1]) for p in fits_dir.glob("KIC_*.fits") ]
    
    # only use those
    koi = koi[koi["kepid"].isin(kids)]


# -----------------------------
# Process the stars, produce summary statistics for simple ML and phase-folded curves for CNN
# -----------------------------

print ("Running over KEPIDs = ",koi["kepid"])

rows = []

with PdfPages(OUTPUT_PDF) as pdf:
    istar = 0
    nstars = len(koi)

    for _, row in koi.iterrows():
        kepid = int(row["kepid"])
        label = 1 if row["koi_disposition"] == "CONFIRMED" else 0

        print(f" === STAR {istar}/{nstars} KIC {kepid} ({row['koi_disposition']})")
        istar += 1

        kic_fname = f"fits/KIC_{kepid}.fits"

        # does the FITS file already exist locally?
        if not os.path.exists(kic_fname):
            print(" - downloading...")
            try:
                search = lk.search_lightcurve(f"KIC {kepid}", mission="Kepler")
                if len(search) == 0:
                    continue
                lc = search.download_all().stitch()
                lc = clean_lc(lc)
                lc.to_fits(kic_fname, overwrite=True)
                print(" - Saved to " + kic_fname)
            except Exception as e:
                print(f"Download failed: {e}")
                continue
        else:
            print(" - Read from " + kic_fname)
            lc = lk.read(kic_fname)

        time = lc.time.value
        flux = lc.flux.value

        print (" - number of samples: ",len(flux))

        # very large curves take forever, we'll just use short ones
        if (len(flux)>250000):
            continue

        # Improved detrending        
        flux, trend = robust_flatten(time, flux, window_length=2.0)

        # Plot cleaned light curve
        fig, ax = plt.subplots(figsize=(8, 3))
        plot_subsample(ax, time, flux, trend)
        ax.set_title(f"KIC {kepid}: cleaned light curve (1% sample)")
        ax.set_xlabel("Time [days]")
        ax.set_ylabel("Normalized flux")
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)

        # BLS period search
        print (" - searching for boxes")
        periods = np.linspace(0.6, 30, 6000)
        durations = np.linspace(0.02, 0.3, 10)
        bls = BoxLeastSquares(time, flux)
        result = bls.power(periods, durations)

        power = result.power
        sde = (power - np.median(power)) / np.std(power)  # Signal Detection Efficiency

        best = np.argmax(sde)
        period = result.period[best]
        t0 = result.transit_time[best]
        duration = result.duration[best]
        depth = result.depth[best]
        snr = sde[best]

        evenodd_ratio = 100.0
        if snr > 6:
            try:
                period, t0, duration, depth = refine_period(time, flux, period, t0)
                stats = bls.compute_stats(period, duration, t0)
                evenodd_ratio = abs(stats['depth_odd'][0] / stats['depth_even'][0]) if stats['depth_even'][0] != 0 else 100.0
            except:
                evenodd_ratio = 100.0

        # Phase folding and plotting
        phase, folded = folded_views(time, flux, period, t0)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(phase, folded, ".", color='black', alpha=0.7)
        ax.set_title(f"Folded light curve (P={period:.3f} d, SDE={snr:.1f}, label={label})")
        ax.set_xlabel("Phase")
        ax.set_ylabel("Flux")
        ax.grid(True, alpha=0.3)
        pdf.savefig(fig)
        plt.close(fig)

        # Save folded candidate
        save_candidate_npz(
            base_dir="candidates",
            kepid=kepid,
            phase=phase,
            flux_folded=folded,
            meta=dict(period=period, t0=t0, duration=duration, depth=depth)
        )

        # Feature extraction
        features = {
            "period": period,
            "depth": depth,
            "snr": snr,
            "evenodd_ratio": evenodd_ratio,
            "folded_std": np.nanstd(folded),
            "folded_min": np.nanmin(folded),
            "folded_ptp": np.nanmax(folded) - np.nanmin(folded),
            "label": label,
            "kepid": kepid
        }
        rows.append(features)

        # -----------------------------
        # Incremental save of features table
        # -----------------------------
        # make a single-row dataframe to append to current file, so that we don't lose it in case of a crash
        df_new = pd.DataFrame([features])  # one row only
        save_file = "kepler_candidates.csv"

        file_exists = os.path.isfile(save_file)

        if file_exists:
            df = pd.read_csv(save_file)
            df = df.drop_duplicates(subset='kepid', keep='first')
            if kepid not in df['kepid'].values:
                tot_df = pd.concat([df, df_new], ignore_index=True)
                tot_df.to_csv(save_file, mode='w', index=False)
        else:
            df_new.to_csv(save_file, mode='w', index=False)

print(f"PDF report written to {OUTPUT_PDF}")
