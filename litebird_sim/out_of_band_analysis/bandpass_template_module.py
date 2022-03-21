import litebird_sim as lbs
import numpy as np
from scipy import signal

class  Band(object) :
    def __init__(self,nu_center , bandwidth=0.2 ,range_offset = 10 ,   nsamp=128  ,  name='band'   ):
        """
        Constructor of the class
        nu_center: center frequency in GHz
        bandwidth :  width of the band (default=0.2)
        range_offset: increase the freq. range by an offset (default +- 10 from the edges)
        nsamp : number of samples (default=128)
        """
        self.nu0 = nu_center
        self.bw = bandwidth
        self.f0, self. f1  = self.get_edges ()

        # we extend the wings of the top-hat bandpass  with 10 samples before  and after the edges
        bandrange =self. f0 -range_offset ,self. f1 +range_offset
        self.freqs  = np.linspace(bandrange[0], bandrange[1], nsamp )
        self.isnormalized = False
        self._name = name



    def get_edges (self ) :
        """
        get the edges of the tophat band
        """
        return self.nu0*(1- self.bw /2),self.nu0*(1+self.bw/2)




    def get_top_hat_bandpass( self  ,
                             normalize= False  , apodization= None  ):
        """
        Sample  a top-hat bandpass, givn the centroid and the bandwidth
        normalize:
        normalize the transmission coefficients so that its integral = 1 over the freq.  band
        apodization:
        if None no apodization is applied to the edges, otherwise a string between `cosine` or `exp` will
        apodize the edges following the chosen  profile
        """


        self. weights = np.zeros_like(self.freqs)
        mask = np.ma.masked_inside( self.freqs,self.f0,self.f1 ).mask
        self. weights [mask] = 1.

        if apodization == 'cosine' :
            print(f"Apodizing w/ {apodization} profile")
            self. cosine_apodize_bandpass()

        elif apodization == 'exp':
            print(f"Apodizing w/ {apodization} profile")
            self. exp_apodize_bandpass()

        elif  apodization   is None:
            print("/!\ Band is not apodized")
        if normalize :
            self.normalize_band()


    def normalize_band(self):
        """
        Normalize the band transmission coefficients

        """
        A = np.trapz(self. weights , self.freqs )
        self. weights  /=A
        self.isnormalized=True



    def exp_apodize_bandpass( self,  alpha = 1, beta = 1):
        """Define a bandpass with exponential tails and unit transmission in band
        freqs: frequency in GHz

        alpha: out-of-band exponential decay index for low freq edge
        beta: out-of-band exponential decay index for high freq edge

        If alpha and beta are not specified a value of 1 is used for both.

        """
        mask_beta=np.ma.masked_greater(self.freqs,self.f1 ).mask
        self.weights[mask_beta] = np.exp(-beta * (self.freqs[mask_beta] - self.f1))
        mask_alpha = np.ma.masked_less( self.freqs, self.f0 ).mask
        self. weights[mask_alpha] = np.exp(alpha * (self.freqs [mask_alpha] -self. f0))

    def cosine_apodize_bandpass( self, a = 5  ):
        """
        Define a bandpass with cosine tails and unit transmission in band
        a:
        is the numerical factor related to the apodization length

        """

        apolength =   self.bw/a
        apod = lambda x, a,b: (1 + np.cos((x-a)/(b-a)  *np.pi ))/2
        f_above= self.nu0 * ( 1 + self.bw/2 + apolength )
        f_below= self.nu0 * ( 1 - self.bw/2 - apolength )
        mask_above=np.ma.masked_inside(self.freqs,self.f1 , f_above  ).mask

        x_ab = np.linspace(self.f1 , f_above,self.freqs[mask_above].size )

        self.weights[mask_above] = apod(x_ab,self.f1 , f_above)

        mask_below=np.ma.masked_inside(self.freqs, f_below, self.f0 ).mask
        x_bel  = np.linspace(f_below, self.f0,self.freqs[mask_below].size )
        self.weights[mask_below] = apod( x_bel, self.f0 , f_below)

    # Chebyshev profile bandpass
    def get_chebyshev_bandpass(self,  order = 3, ripple_dB = 3, normalize=False ):
        """
        Define a bandpass with chebyshev prototype
        order: chebyshev filter order
        ripple_dB: maximum ripple amplitude in decibels

        If order and ripple_dB are not specified a value of 3 is used for both.

        """
        b, a = signal.cheby1(order, ripple_dB, [2.*np.pi*self.f0*1e9, 2.*np.pi*self.f1*1e9], 'bandpass', analog=True)
        w, h = signal.freqs(b, a, worN=self.freqs*2*np.pi*1e9)

        self.weights = abs(h)
        if normalize :
            A = self.get_normalization()
            self. weights  /=A
            self.isnormalized=True


    def get_normalization (self):
        """
        Estimate the integral over the frequency band
        """
        return np.trapz(self.weights,self.freqs )


    # Find effective central frequency of a bandpass profile
    def find_central_frequency(self):
        """Find the effective central frequency of
        a bandpass profile as defined in https://arxiv.org/abs/1303.5070
        """
        if self.isnormalized:
            return np.trapz(self.freqs*self.weights,self.freqs )
        else :
            return np.trapz (self.freqs*self.weights ,self.freqs)/self.get_normalization()


    def interpolate_band  (self ):
        """
        This function aims at building the sampler in order to generate random samples
        statistically equivalent to the model bandpass
        """
        #normalize band

        if not  self.isnormalized:
            self.normalize_band()
        #Interpolate the band
        b = sp.interpolate.interp1d(x=self.freqs , y=self.weights )
        #estimate the CDF
        Pnu =np.array([sp.integrate.quad(b , a =self.freqs.min(), b=inu  )[0]  for inu in self.freqs[1:] ])
        #interpolate the inverse CDF
        self.Sampler = sp.interpolate.interp1d(Pnu ,self.freqs[:-1] + np.diff(self.freqs),
                                               bounds_error=False, fill_value="extrapolate")

    def bandpass_resampling(self,  bstrap_size= 1000, nresample=54 , model =None  ):
        """
        Resample a  bandpass with bootstrap resampling.
        Notice that the user can provide any sampler built with the `interpolate_band`
        method, if not provided an error will be raised!
        bstrap_size : int
        encodes the size of the random dataset  to be generated from the Sampler
        nresample :
        define how fine is the grid for the resampled bandpass
        """

        if model is not  None :
            print(f"Sampler  from {model._name }")
            Sampler= model.Sampler
        else:
            try :
                Sampler = self.Sampler
            except AttributeError :
                print("Can't resample if no sampler is built and/or provided, interpolating the band")
                self. interpolate_band()
                Sampler= self.Sampler



        X =  np.random.uniform(size=bstrap_size)
        bins_nu=np.linspace(self.freqs.min(), self.freqs.max(),nresample)
        h, xb =np.histogram(Sampler( X ), density=True ,bins= bins_nu   )

        nu_b  = xb[:-1] + np.diff(xb)
        resampled_bpass =abs(sp.interpolate.interp1d(nu_b, h, kind='cubic', bounds_error=False, fill_value="extrapolate")(self.freqs))
        if self.isnormalized:
            return   resampled_bpass/np.trapz(resampled_bpass,self.freqs )
        else:
            return resampled_bpass





# Chebyshev profile lowpass
def lowpass_chebyshev(freqs, f0, order = 1, ripple_dB = 1):
    """Define a lowpass with chebyshev prototype
    freqs: frequency in GHz
    f0: low-frequency edge of the band in GHz
    order: chebyshev filter order
    ripple_dB: maximum ripple amplitude in decibels

    If order and ripple_dB are not specified a value of 3 is used for both.

    """
    b, a = signal.cheby1(order, ripple_dB, 2.*np.pi*f0*1e9, 'lowpass', analog=True)
    w, h = signal.freqs(b, a, worN=freqs*2*np.pi*1e9)

    transmission = abs(h)

    return transmission

# Find effective central frequency of a bandpass profile
def find_central_frequency(freqs, bandpass):
    """Find the effective central frequency of
    a bandpass profile as defined in https://arxiv.org/abs/1303.5070
    freqs: frequency in GHz
    bandpass: transmission profile
    """
    df = freqs[1]-freqs[0]

    fc = sum(freqs*bandpass*df)/sum(bandpass*df)

    return fc

# Add high frequency leakage to a bandpass profile
def add_high_frequency_transmission(freqs, bandpass, location = 3, transmission = 0.5):
    """Add high frequency leakage
    freqs: frequency in GHz
    bandpass: transmission profile
    location: multiple of the central frequency of the bandpass profile where add the leakage
    transmission: relative amplitude of the high frequency leakage with respect to the nominal band

    If location and transmission are not specified a value of 3 and 0.5 are set by default.

    """

    df = freqs[1]-freqs[0]
    fc = find_central_frequency(freqs, bandpass)

    diff_freq = abs(freqs-fc)
    i_fc = np.where(diff_freq == min(diff_freq))[0]
    delta_fc = abs(freqs[-1] - freqs[i_fc])

    high_freq_fc = location*fc

    new_freqs_min = freqs[0]
    new_freqs_max = high_freq_fc + delta_fc

    freqs_new = np.linspace(freqs[0], new_freqs_max, int((new_freqs_max-new_freqs_min)/df + 1))
    bandpass_new = np.zeros_like(freqs_new)

    for i in range(len(freqs_new)):

        if i < len(freqs):

            bandpass_new[i] = bandpass[i]

        elif i >= (location-1)*i_fc:

            bandpass_new[i] = transmission*bandpass[i-int((location-1)*i_fc)]

    return freqs_new, bandpass_new
