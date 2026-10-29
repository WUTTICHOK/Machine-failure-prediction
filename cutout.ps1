param($src, $dst, [int]$maxSize = 420)
Add-Type -AssemblyName System.Drawing
Add-Type -ReferencedAssemblies System.Drawing -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

public static class Cutout {
    static int Lum(byte[] p, int i) { return (p[i] * 114 + p[i + 1] * 587 + p[i + 2] * 299) / 1000; } // BGRA

    public static Bitmap Run(Bitmap input, int darkMax, int whiteMin, int maxSize) {
        int w = input.Width, h = input.Height;
        Bitmap bmp = new Bitmap(w, h, PixelFormat.Format32bppArgb);
        using (Graphics g = Graphics.FromImage(bmp)) g.DrawImage(input, 0, 0, w, h);
        Rectangle r = new Rectangle(0, 0, w, h);
        BitmapData d = bmp.LockBits(r, ImageLockMode.ReadWrite, PixelFormat.Format32bppArgb);
        int stride = d.Stride;
        byte[] px = new byte[stride * h];
        Marshal.Copy(d.Scan0, px, 0, px.Length);

        // 1) flood fill the dark checkerboard from the image border; the white outline stops it
        bool[] bg = new bool[w * h];
        Queue<int> q = new Queue<int>();
        Action<int, int> seed = (x, y) => {
            int k = y * w + x;
            if (!bg[k] && Lum(px, y * stride + x * 4) < darkMax) { bg[k] = true; q.Enqueue(k); }
        };
        for (int x = 0; x < w; x++) { seed(x, 0); seed(x, h - 1); }
        for (int y = 0; y < h; y++) { seed(0, y); seed(w - 1, y); }
        int[] dx = { 1, -1, 0, 0 }, dy = { 0, 0, 1, -1 };
        while (q.Count > 0) {
            int k = q.Dequeue(), x = k % w, y = k / w;
            for (int n = 0; n < 4; n++) {
                int nx = x + dx[n], ny = y + dy[n];
                if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                int nk = ny * w + nx;
                if (bg[nk]) continue;
                if (Lum(px, ny * stride + nx * 4) < darkMax) { bg[nk] = true; q.Enqueue(nk); }
            }
        }
        // 2) background -> transparent; grey fringe next to it -> white with partial alpha
        for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
            int k = y * w + x, i = y * stride + x * 4;
            if (bg[k]) { px[i + 3] = 0; continue; }
            bool nearBg = false;
            for (int yy = Math.Max(0, y - 2); yy <= Math.Min(h - 1, y + 2) && !nearBg; yy++)
                for (int xx = Math.Max(0, x - 2); xx <= Math.Min(w - 1, x + 2); xx++)
                    if (bg[yy * w + xx]) { nearBg = true; break; }
            if (!nearBg) continue;
            int l = Lum(px, i);
            if (l < whiteMin) {
                int a = Math.Max(0, Math.Min(255, (l - darkMax) * 255 / Math.Max(1, whiteMin - darkMax)));
                px[i] = 255; px[i + 1] = 255; px[i + 2] = 255; px[i + 3] = (byte)a;
            }
        }
        Marshal.Copy(px, 0, d.Scan0, px.Length);
        bmp.UnlockBits(d);

        // 3) crop to the sticker and resize
        int minX = w, minY = h, maxX = -1, maxY = -1;
        for (int y = 0; y < h; y++) for (int x = 0; x < w; x++)
            if (px[y * stride + x * 4 + 3] > 10) {
                if (x < minX) minX = x; if (x > maxX) maxX = x;
                if (y < minY) minY = y; if (y > maxY) maxY = y;
            }
        int pad = 4;
        minX = Math.Max(0, minX - pad); minY = Math.Max(0, minY - pad);
        maxX = Math.Min(w - 1, maxX + pad); maxY = Math.Min(h - 1, maxY + pad);
        int cw = maxX - minX + 1, ch = maxY - minY + 1;
        double s = Math.Min(1.0, (double)maxSize / Math.Max(cw, ch));
        int ow = (int)Math.Round(cw * s), oh = (int)Math.Round(ch * s);
        Bitmap outBmp = new Bitmap(ow, oh, PixelFormat.Format32bppArgb);
        using (Graphics g = Graphics.FromImage(outBmp)) {
            g.Clear(Color.Transparent);
            g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
            g.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.HighQuality;
            g.CompositingMode = System.Drawing.Drawing2D.CompositingMode.SourceCopy;
            g.DrawImage(bmp, new Rectangle(0, 0, ow, oh), new Rectangle(minX, minY, cw, ch), GraphicsUnit.Pixel);
        }
        bmp.Dispose();
        return outBmp;
    }
}
'@
$img = [System.Drawing.Bitmap]::FromFile($src)
$out = [Cutout]::Run($img, 110, 225, $maxSize)
$img.Dispose()
$out.Save($dst, [System.Drawing.Imaging.ImageFormat]::Png)
"{0} x {1}" -f $out.Width, $out.Height
$out.Dispose()
